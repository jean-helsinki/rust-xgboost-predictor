use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, BufRead, Result};
use std::collections::LinkedList;

use crate::common::types::*;


pub fn get_resource(rel_path: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/resources");
    d.push(rel_path);
    d
}

pub fn open_resource_file(rel_path: &str) -> Result<File> {
    File::open(get_resource(rel_path))
}

pub fn load_data(rel_path: &str) -> LinkedList<DataItem> {
    let mut file = open_resource_file(rel_path).unwrap();
    let reader = BufReader::new(file);
    let mut result = LinkedList::<DataItem>::new();

    for line in reader.lines() {
        let dataline = line.unwrap();
        let values: Vec<&str> = dataline.split(' ').collect();
        let mut map = TestMap::new();
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

pub fn load_expectation(rel_path: &str) -> LinkedList<f32> {
    let mut file = open_resource_file(rel_path).unwrap();
    let reader = BufReader::new(file);
    let mut result = LinkedList::<f32>::new();
    for line in reader.lines() {
        let dataline = line.unwrap();
        result.push_back(dataline.parse::<f32>().unwrap());
    };
    result
}