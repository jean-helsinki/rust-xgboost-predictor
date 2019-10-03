
trait ObjFunction {
    fn predTransformVec(preds: Vec<f64>) -> Vec<f64>;
    fn predTransform(pred: f64) -> f64;
    fn sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }
}



struct RegLossObjLogistic {}

impl ObjFunction for RegLossObjLogistic {
    /// Logistic regression.
    fn predTransformVec(preds: Vec<f64>) -> Vec<f64> {
        return preds.into_iter().map(RegLossObjLogistic::sigmoid).collect();
    }
    fn predTransform(pred: f64) -> f64 {
        RegLossObjLogistic::sigmoid(pred)
    }
}

struct SoftmaxMultiClassObjClassify {}

impl ObjFunction for SoftmaxMultiClassObjClassify {
    /// Multiclass classification.
    fn predTransformVec(preds: Vec<f64>) -> Vec<f64> {
        let (maxIndex, max) = preds.iter().enumerate().max_by(|(i1, x1), (i2, x2)| x1.partial_cmp(x2).unwrap()).unwrap();
        return vec![maxIndex as f64;1]
    }
    fn predTransform(pred: f64) -> f64 {
        std::unimplemented!();
    }
}

struct SoftmaxMultiClassObjProb {}

impl ObjFunction for SoftmaxMultiClassObjProb {
    ///  Multiclass classification (predicted probability).
    fn predTransformVec(preds: Vec<f64>) -> Vec<f64> {
        let max = preds.iter().fold(preds.get(0), |a, b| b.max(*a.unwrap())).unwrap();
        let sum = preds.iter().map(|x| (x - max).exp()).sum();
        return preds.iter().map(|x| (x - max).exp()/sum).collect();
    }
    fn predTransform(pred: f64) -> f64 {
        std::unimplemented!();
    }
}

