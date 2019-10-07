use crate::model_reader::{ModelReader, ModelReadResult};
use std::f32;


struct Param {
    /// number of start root
    num_roots: i32,
    /// total number of nodes
    num_nodes: i32,
    /// number of deleted nodes
    num_deleted: i32,
    /// maximum depth, this is a statistics of the tree
    max_depth: i32,
    /// number of features used for tree construction
    num_feature: i32,
    /// leaf vector size, used for vector tree used to store more than one dimensional information in tree
    size_leaf_vector: i32,
    /// reserved part
    reserved: Vec<i32>,
}

impl Param {
    fn new<T: ModelReader>(reader: &mut T) -> ModelReadResult<Param> {
        return Ok(Param {
            num_roots: reader.read_i32_le()?,
            num_nodes: reader.read_i32_le()?,
            num_deleted: reader.read_i32_le()?,
            max_depth: reader.read_i32_le()?,
            num_feature: reader.read_i32_le()?,
            size_leaf_vector: reader.read_i32_le()?,
            reserved: reader.read_int_vec(31)?,
        })
    }
}


struct Node {
    /// pointer to parent, highest bit is used to indicate whether it's a left child or not
    parent: i32,
    /// pointer to right
    cleft: i32,
    /// pointer to right
    cright: i32,
    /// split feature index, left split or right split depends on the highest bit
    leaf_value: f32,
    split_cond: f32,
    default_next: i32,
    split_index: i32,
    is_leaf: bool,
}

impl Node {
    fn is_default_left(sindex: i32) -> bool {
        return (sindex >> 31) != 0;
    }

    fn decode_split_index(sindex: i32) -> i32 {
        return ((sindex as i64) & ((1i64 << 31) - 1i64)) as i32;
    }

    fn new<T: ModelReader>(reader: &mut T) -> ModelReadResult<Node> {
        let (parent, cleft, cright, sindex) =
            (reader.read_i32_le()?, reader.read_i32_le()?, reader.read_i32_le()?, reader.read_i32_le()?);
        let is_leaf = cleft == -1;

        let leaf_value = if is_leaf {reader.read_float_le()?} else { f32::NAN };
        let split_cond = if is_leaf { f32::NAN } else {reader.read_float_le()?};

        let split_index = Node::decode_split_index(sindex);
        let default_next = if Node::is_default_left(split_index) {cleft} else {cright};

        return Ok(Node {
            parent: parent,
            cleft: cleft,
            cright: cright,
            leaf_value: leaf_value,
            split_cond: split_cond,
            default_next: default_next,
            split_index: split_index,
            is_leaf: is_leaf,
        })

    }
}


struct RTreeNodeStat {
    /// loss chg caused by current split
    loss_chg: f32,
    /// sum of hessian values, used to measure coverage of data
    sum_hess: f32,
    /// weight of current node
    base_weight: f32,
    /// number of child that is leaf node known up to now
    leaf_child_cnt: i32,
}

impl RTreeNodeStat {
    fn new<T: ModelReader>(reader: &mut T) -> ModelReadResult<RTreeNodeStat> {
        return Ok(RTreeNodeStat{
            loss_chg: reader.read_float_le()?,
            sum_hess: reader.read_float_le()?,
            base_weight: reader.read_float_le()?,
            leaf_child_cnt: reader.read_i32_le()?,
        })
    }
}

/// Regression tree
pub struct RegTree {
    param: Param,
    nodes: Vec<Node>,
    stats: Vec<RTreeNodeStat>,
}

impl RegTree {
    fn new<T: ModelReader>(reader: &mut T) -> ModelReadResult<RegTree> {
        let param = Param::new(reader)?;
        let nodes: ModelReadResult<Vec<Node>> = (0..param.num_nodes).map(|_| Node::new(reader)).collect();
        let stats: ModelReadResult<Vec<RTreeNodeStat>> = (0..param.num_nodes).map(|_| RTreeNodeStat::new(reader)).collect();
        return Ok(RegTree{param, nodes: nodes?, stats: stats?});
    }
}
