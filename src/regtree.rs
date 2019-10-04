use crate::model_reader::ModelReader;

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
    fn new<T: ModelReader>(reader: &mut T) -> Param {
        return Param {
            num_roots: reader.read_int_le(),
            num_nodes: reader.read_int_le(),
            num_deleted: reader.read_int_le(),
            max_depth: reader.read_int_le(),
            num_feature: reader.read_int_le(),
            size_leaf_vector: reader.read_int_le(),
            reserved: reader.read_int_array(31),

        }
    }
}