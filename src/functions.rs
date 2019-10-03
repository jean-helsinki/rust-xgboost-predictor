
pub enum FunctionType {
    RankPairwise,
    BinaryLogistic,
    BinaryLogitraw,
    MultiSoftmax,
    MultiSoftprob,
    RegLinear,
    RegLogistic,
}

pub trait ObjFunction {
    fn pred_transform_vec(preds: Vec<f64>) -> Vec<f64>;
    fn pred_pransform(pred: f64) -> f64;
    fn sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    fn from_type(tp: FunctionTypes) -> Box<ObjFunction> {

    }
}

struct RegLossObjLogistic {}

impl ObjFunction for RegLossObjLogistic {
    /// Logistic regression.
    fn pred_transform_vec(preds: Vec<f64>) -> Vec<f64> {
        return preds.into_iter().map(RegLossObjLogistic::sigmoid).collect();
    }
    fn pred_pransform(pred: f64) -> f64 {
        RegLossObjLogistic::sigmoid(pred)
    }
}

struct SoftmaxMultiClassObjClassify {}

impl ObjFunction for SoftmaxMultiClassObjClassify {
    /// Multiclass classification.
    fn pred_transform_vec(preds: Vec<f64>) -> Vec<f64> {
        match preds.get(0) {
            Option::Some(init) => {
                let (max_index, _max) = preds.iter().enumerate()
                    .fold((0, init), |(i1, v1), (i2, v2)|
                        if v1 >= v2 { (i1, v1) } else { (i2, v2) });
                return vec![max_index as f64; 1]
            }
            // empty vector
            Option::None => preds
        }
    }
    fn pred_pransform(_pred: f64) -> f64 {
        std::unimplemented!();
    }
}

struct SoftmaxMultiClassObjProb {}

impl ObjFunction for SoftmaxMultiClassObjProb {
    ///  Multiclass classification (predicted probability).
    fn pred_transform_vec(preds: Vec<f64>) -> Vec<f64> {
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
    fn pred_pransform(_pred: f64) -> f64 {
        std::unimplemented!();
    }
}
