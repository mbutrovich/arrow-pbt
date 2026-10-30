//! Predicates on `DataType` that the properties and case strategies share

use arrow_schema::{DataType, UnionMode};

/// Whether `dt` or any type nested in it satisfies `f`
///
/// # Example
///
/// ```
/// use arrow_pbt::types::{any_type, is_view};
/// use arrow_schema::{DataType, Field};
/// use std::sync::Arc;
/// let t = DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)));
/// assert!(any_type(&t, &is_view));
/// ```
pub fn any_type(dt: &DataType, f: &dyn Fn(&DataType) -> bool) -> bool {
    f(dt)
        || match dt {
            DataType::List(c)
            | DataType::LargeList(c)
            | DataType::ListView(c)
            | DataType::LargeListView(c)
            | DataType::FixedSizeList(c, _)
            | DataType::Map(c, _) => any_type(c.data_type(), f),
            DataType::Struct(fs) => fs.iter().any(|x| any_type(x.data_type(), f)),
            DataType::Union(fs, _) => fs.iter().any(|(_, x)| any_type(x.data_type(), f)),
            DataType::Dictionary(_, v) => any_type(v, f),
            DataType::RunEndEncoded(_, v) => any_type(v.data_type(), f),
            _ => false,
        }
}

/// `Utf8View` and `BinaryView`
///
/// # Example
///
/// ```
/// use arrow_pbt::types::is_view;
/// use arrow_schema::DataType;
/// assert!(is_view(&DataType::BinaryView));
/// assert!(!is_view(&DataType::Binary));
/// ```
pub fn is_view(dt: &DataType) -> bool {
    matches!(dt, DataType::Utf8View | DataType::BinaryView)
}

/// A dictionary of `Utf8View` or `BinaryView` values
///
/// # Example
///
/// ```
/// use arrow_pbt::types::is_view_dict;
/// use arrow_schema::DataType;
/// let t = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8View));
/// assert!(is_view_dict(&t));
/// ```
pub fn is_view_dict(dt: &DataType) -> bool {
    matches!(dt, DataType::Dictionary(_, v) if is_view(v))
}

/// A union in sparse mode
///
/// # Example
///
/// ```
/// use arrow_pbt::types::is_sparse_union;
/// use arrow_schema::{DataType, UnionFields, UnionMode};
/// assert!(is_sparse_union(&DataType::Union(UnionFields::empty(), UnionMode::Sparse)));
/// assert!(!is_sparse_union(&DataType::Union(UnionFields::empty(), UnionMode::Dense)));
/// ```
pub fn is_sparse_union(dt: &DataType) -> bool {
    matches!(dt, DataType::Union(_, UnionMode::Sparse))
}
