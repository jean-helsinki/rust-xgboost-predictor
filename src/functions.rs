pub enum FunctionType {
    RankPairwise,
    BinaryLogistic,
    BinaryLogitraw,
    MultiSoftmax,
    MultiSoftprob,
    RegLinear,
}

pub struct ObjFunction {
    pub vector: fn(&Vec<f64>) -> Vec<f64>,
    pub scalar: fn(f64) -> f64,
}

fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}

fn dump_vec(preds: &Vec<f64>) -> Vec<f64> {
    return preds.clone();
}

fn dump(pred: f64) -> f64 {
    return pred;
}

/// Logistic regression.
fn logistic_vec(preds: &Vec<f64>) -> Vec<f64> {
    return preds.into_iter().map(|x| sigmoid(*x)).collect();
}

/// Multiclass classification.
fn multiclass_vec(preds: &Vec<f64>) -> Vec<f64> {
    match preds.get(0) {
        Option::Some(init) => {
            let (max_index, _max) = preds.iter().enumerate()
                .fold((0, init), |(i1, v1), (i2, v2)|
                    if v1 >= v2 { (i1, v1) } else { (i2, v2) });
            return vec![max_index as f64; 1];
        }
        // empty vector
        Option::None => preds.clone()
    }
}

fn unimplemented(_pred: f64) -> f64 {
    std::unimplemented!();
}

///  Multiclass classification (predicted probability).
fn multiclass_pred_prob_vec(preds: &Vec<f64>) -> Vec<f64> {
    match preds.get(0) {
        Option::Some(init) => {
            let max = preds.iter().fold(*init, |a, b| b.max(a));
            let sum: f64 = preds.iter().map(|x| (x - max).exp()).sum();
            return preds.iter().map(|x| (x - max).exp() / sum).collect();
        }
        // empty vector
        Option::None => preds.clone()
    }
}


pub fn get_classify_function(tp: FunctionType) -> ObjFunction {
    match tp {
        FunctionType::RankPairwise | FunctionType::BinaryLogitraw | FunctionType::RegLinear
        => ObjFunction { vector: dump_vec, scalar: dump },
        FunctionType::BinaryLogistic => ObjFunction { vector: logistic_vec, scalar: sigmoid },
        FunctionType::MultiSoftmax => ObjFunction { vector: multiclass_vec, scalar: unimplemented },
        FunctionType::MultiSoftprob => ObjFunction { vector: multiclass_pred_prob_vec, scalar: unimplemented },
    }
}

pub fn get_classify_func_type(obj_name: &[u8]) -> Option<FunctionType> {
    return match obj_name {
        b"rank:pairwise" => Some(FunctionType::RankPairwise),
        b"binary:logistic" => Some(FunctionType::BinaryLogistic),
        b"multi:softmax" => Some(FunctionType::MultiSoftmax),
        b"multi:softprob" => Some(FunctionType::MultiSoftprob),
        b"reg:linear" => Some(FunctionType::RegLinear),
        _ => None,
    };
}

#[cfg(test)]
mod tests {
    use crate::functions::get_classify_function;
    use crate::functions::FunctionType::BinaryLogistic;

    #[test]
    fn it_works() {
        let func = get_classify_function(BinaryLogistic);
        (func.vector)(&vec![1.0f64, 4.6f64]);
    }
}
