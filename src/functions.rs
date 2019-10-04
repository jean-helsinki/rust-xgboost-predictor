pub enum FunctionType {
    RankPairwise,
    BinaryLogistic,
    BinaryLogitraw,
    MultiSoftmax,
    MultiSoftprob,
    RegLinear,
}

pub struct ClassifyFunction {
    vector: fn(Vec<f64>) -> Vec<f64>,
    scalar: fn(f64) -> f64,
}

fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}

fn dump_vec(preds: Vec<f64>) -> Vec<f64> {
    return preds;
}

fn dump(pred: f64) -> f64 {
    return pred;
}

/// Logistic regression.
fn logistic_vec(preds: Vec<f64>) -> Vec<f64> {
    return preds.into_iter().map(sigmoid).collect();
}

/// Multiclass classification.
fn multiclass_vec(preds: Vec<f64>) -> Vec<f64> {
    match preds.get(0) {
        Option::Some(init) => {
            let (max_index, _max) = preds.iter().enumerate()
                .fold((0, init), |(i1, v1), (i2, v2)|
                    if v1 >= v2 { (i1, v1) } else { (i2, v2) });
            return vec![max_index as f64; 1];
        }
        // empty vector
        Option::None => preds
    }
}

fn unimplemented(_pred: f64) -> f64 {
    std::unimplemented!();
}

///  Multiclass classification (predicted probability).
fn multiclass_pred_prob_vec(preds: Vec<f64>) -> Vec<f64> {
    match preds.get(0) {
        Option::Some(init) => {
            let max = preds.iter().fold(*init, |a, b| b.max(a));
            let sum: f64 = preds.iter().map(|x| (x - max).exp()).sum();
            return preds.iter().map(|x| (x - max).exp() / sum).collect();
        }
        // empty vector
        Option::None => preds
    }
}


pub fn get_classify_function(tp: FunctionType) -> ClassifyFunction {
    match tp {
        FunctionType::RankPairwise | FunctionType::BinaryLogitraw | FunctionType::RegLinear
        => ClassifyFunction { vector: dump_vec, scalar: dump },
        FunctionType::BinaryLogistic => ClassifyFunction { vector: logistic_vec, scalar: sigmoid },
        FunctionType::MultiSoftmax => ClassifyFunction { vector: multiclass_vec, scalar: unimplemented },
        FunctionType::MultiSoftprob => ClassifyFunction { vector: multiclass_pred_prob_vec, scalar: unimplemented },
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::get_classify_function;
    use crate::functions::FunctionType::BinaryLogistic;

    #[test]
    fn it_works() {
        let func = get_classify_function(BinaryLogistic);
        (func.vector)(vec![1.0f64, 4.6f64]);
    }
}
