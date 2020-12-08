use crate::common::types::*;

pub struct PredictionTask {
    pub expectation_suffix: String,
    pub predict: Box<dyn Fn(&TestPredictor, &TestMap) -> Vec<f32>>,
}

pub fn predict() -> PredictionTask {
    PredictionTask {
        expectation_suffix: "predict".to_string(),
        predict: Box::new(|predictor: &TestPredictor, map: &TestMap| {
            predictor.predict(map, false, 0)
        }),
    }
}

pub fn predict_margin() -> PredictionTask {
    PredictionTask {
        expectation_suffix: "margin".to_string(),
        predict: Box::new(|predictor: &TestPredictor, map: &TestMap| {
            predictor.predict(map, true, 0)
        }),
    }
}

pub fn predict_with_ntree_limit(n: usize) -> PredictionTask {
    PredictionTask {
        expectation_suffix: "predict_ntree".to_string(),
        predict: Box::new(move |predictor: &TestPredictor, map: &TestMap| {
            predictor.predict(map, false, n)
        }),
    }
}

pub fn predict_with_excessive_ntree_limit() -> PredictionTask {
    PredictionTask {
        expectation_suffix: "predict".to_string(),
        predict: Box::new(|predictor: &TestPredictor, map: &TestMap| {
            predictor.predict(map, false, 1000)
        }),
    }
}

pub fn predict_single() -> PredictionTask {
    PredictionTask {
        expectation_suffix: "predict".to_string(),
        predict: Box::new(|predictor: &TestPredictor, map: &TestMap| {
            vec![predictor.predict_single(map, false, 0)]
        }),
    }
}

pub fn predict_leaf() -> PredictionTask {
    PredictionTask {
        expectation_suffix: "leaf".to_string(),
        predict: Box::new(|predictor: &TestPredictor, map: &TestMap| {
            predictor
                .predict_leaf(map, 0)
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }),
    }
}

pub fn predict_leaf_with_ntree(n: usize) -> PredictionTask {
    PredictionTask {
        expectation_suffix: "leaf_ntree".to_string(),
        predict: Box::new(move |predictor: &TestPredictor, map: &TestMap| {
            predictor
                .predict_leaf(map, n)
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }),
    }
}
