use std::path::PathBuf;
use std::fs::File;

use std::collections::{HashMap, LinkedList};
use xgboost_predictor::fvec::FVecMap;
use std::io::{BufReader, BufRead};
use std::io;
use xgboost_predictor::predictor::Predictor;


fn get_resource(rel_path: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/resources");
    d.push(rel_path);
    d
}

fn open_resource_file(rel_path: &str) -> io::Result<File> {
    File::open(get_resource(rel_path))
}

type DataItem = (usize, FVecMap<f32>);

fn load_data(rel_path: &str) -> LinkedList<DataItem> {
    let mut file = open_resource_file(rel_path).unwrap();
    let reader = BufReader::new(file);
    let mut result = LinkedList::<DataItem>::new();

    for line in reader.lines() {
        let dataline = line.unwrap();
        let values: Vec<&str> = dataline.split(' ').collect();
        let mut map = FVecMap::<f32>::new();
        let val = values[0].parse::<usize>().unwrap();
        for s in values[1..].iter() {
            let pair: Vec<&str> = s.split(':').collect();
            map.insert(pair[0].parse::<usize>().unwrap(),
                       pair[1].parse::<f32>().unwrap());
        }
        result.push_back((val, map));
    }
    result
}

#[test]
fn test_predict() {
    let data = load_data("data/agaricus.txt.0.test");
    let mut model_file = open_resource_file("model/gbtree/v47/binary-logistic.model").unwrap();
    let predictor: Predictor<FVecMap<f32>> = Predictor::read_from::<File>(&mut model_file).unwrap();
}