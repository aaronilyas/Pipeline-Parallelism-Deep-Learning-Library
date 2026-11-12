//! Pipeline schedule: local forward/backward op order for one stage rank.

use linalg_core::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Forward(u32),
    Backward(u32),
}

/// Local 1F1B-style order for one pipeline rank.
///
/// Returns [`Error::Schedule`] when `stages` is 0, `rank >= stages`,
/// `microbatches` is 0, or `microbatches > u32::MAX`.
pub fn local_order(stages: usize, rank: usize, microbatches: usize) -> Result<Vec<Op>, Error> {
    if stages == 0 || rank >= stages || microbatches == 0 || microbatches > u32::MAX as usize {
        return Err(Error::Schedule);
    }
    let m = microbatches;
    let w = stages - rank - 1;
    let warmup = w.min(m);
    let steady = m - warmup;
    let mut out = Vec::with_capacity(2 * m);
    for i in 0..warmup {
        out.push(Op::Forward(i as u32));
    }
    for i in 0..steady {
        out.push(Op::Forward((warmup + i) as u32));
        out.push(Op::Backward(i as u32));
    }
    for i in steady..m {
        out.push(Op::Backward(i as u32));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(i: u32) -> Op {
        Op::Forward(i)
    }

    fn b(i: u32) -> Op {
        Op::Backward(i)
    }

    #[test]
    fn s1_r0_m3() {
        assert_eq!(local_order(1, 0, 3).unwrap(), vec![f(0), b(0), f(1), b(1), f(2), b(2)]);
    }

    #[test]
    fn s2_r0_m4() {
        assert_eq!(
            local_order(2, 0, 4).unwrap(),
            vec![f(0), f(1), b(0), f(2), b(1), f(3), b(2), b(3)]
        );
    }

    #[test]
    fn s2_r1_m4() {
        assert_eq!(
            local_order(2, 1, 4).unwrap(),
            vec![f(0), b(0), f(1), b(1), f(2), b(2), f(3), b(3)]
        );
    }

    #[test]
    fn s3_r0_m4() {
        assert_eq!(
            local_order(3, 0, 4).unwrap(),
            vec![f(0), f(1), f(2), b(0), f(3), b(1), b(2), b(3)]
        );
    }

    #[test]
    fn s3_r1_m4() {
        assert_eq!(
            local_order(3, 1, 4).unwrap(),
            vec![f(0), f(1), b(0), f(2), b(1), f(3), b(2), b(3)]
        );
    }

    #[test]
    fn s3_r2_m4() {
        assert_eq!(
            local_order(3, 2, 4).unwrap(),
            vec![f(0), b(0), f(1), b(1), f(2), b(2), f(3), b(3)]
        );
    }

    #[test]
    fn s3_r0_m1() {
        assert_eq!(local_order(3, 0, 1).unwrap(), vec![f(0), b(0)]);
    }

    #[test]
    fn s4_r0_m2() {
        assert_eq!(local_order(4, 0, 2).unwrap(), vec![f(0), f(1), b(0), b(1)]);
    }

    #[test]
    fn rejects_empty_stages_rank_and_microbatches() {
        assert_eq!(local_order(0, 0, 1), Err(Error::Schedule));
        assert_eq!(local_order(2, 2, 1), Err(Error::Schedule));
        assert_eq!(local_order(2, 3, 1), Err(Error::Schedule));
        assert_eq!(local_order(3, 0, 0), Err(Error::Schedule));
        assert_eq!(local_order(1, 0, (u32::MAX as usize) + 1), Err(Error::Schedule));
    }

    #[test]
    fn counts_and_relative_order_hold_for_stages_1_to_5_and_microbatches_1_to_8() {
        for stages in 1..=5 {
            for rank in 0..stages {
                for m in 1..=8 {
                    let order = local_order(stages, rank, m).unwrap();
                    let forwards: Vec<u32> = order
                        .iter()
                        .filter_map(|op| match op {
                            Op::Forward(i) => Some(*i),
                            Op::Backward(_) => None,
                        })
                        .collect();
                    let backwards: Vec<u32> = order
                        .iter()
                        .filter_map(|op| match op {
                            Op::Backward(i) => Some(*i),
                            Op::Forward(_) => None,
                        })
                        .collect();
                    assert_eq!(forwards.len(), m, "stages={stages} rank={rank} m={m}");
                    assert_eq!(backwards.len(), m, "stages={stages} rank={rank} m={m}");
                    assert_eq!(
                        forwards,
                        (0..m as u32).collect::<Vec<_>>(),
                        "stages={stages} rank={rank} m={m}"
                    );
                    assert_eq!(
                        backwards,
                        (0..m as u32).collect::<Vec<_>>(),
                        "stages={stages} rank={rank} m={m}"
                    );
                    for mb in 0..m as u32 {
                        let f_pos = order.iter().position(|op| *op == Op::Forward(mb)).unwrap();
                        let b_pos = order.iter().position(|op| *op == Op::Backward(mb)).unwrap();
                        assert!(
                            f_pos < b_pos,
                            "F({mb}) must precede B({mb}); stages={stages} rank={rank} m={m}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn one_f_one_b_interleaves_when_there_is_enough_work() {
        for stages in 1..=5 {
            for rank in 0..stages {
                let w = stages - rank - 1;
                // Steady length is M - min(w, M). An F follows the first B only when
                // steady >= 2, i.e. M > w + 1.
                for m in (w + 2)..=8 {
                    let order = local_order(stages, rank, m).unwrap();
                    let first_b = order
                        .iter()
                        .position(|op| matches!(op, Op::Backward(_)))
                        .expect("at least one backward");
                    let has_f_after =
                        order[first_b + 1..].iter().any(|op| matches!(op, Op::Forward(_)));
                    assert!(
                        has_f_after,
                        "expected some F after first B when M > w+1; stages={stages} rank={rank} m={m}"
                    );
                }
            }
        }
    }
}
