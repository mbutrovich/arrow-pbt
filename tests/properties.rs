//! One test per property. `ARROW_PBT_CASES`, `ARROW_PBT_SEEDS` and
//! `ARROW_PBT_SHRINK` set the run size (see `arrow_pbt::run`).
//! A failure prints the shrunk input and the command that reruns it.

/// arrow-select
mod arrow_select {
    use arrow_pbt::registry::check;

    #[test]
    fn take() {
        check("take");
    }

    #[test]
    fn filter() {
        check("filter");
    }

    #[test]
    fn interleave() {
        check("interleave");
    }

    #[test]
    fn concat() {
        check("concat");
    }

    #[test]
    fn shift() {
        check("shift");
    }

    #[test]
    fn union_extract() {
        check("union_extract");
    }

    #[test]
    fn zip() {
        check("zip");
    }

    #[test]
    fn merge() {
        check("merge");
    }

    #[test]
    fn merge_n() {
        check("merge_n");
    }

    #[test]
    fn nullif() {
        check("nullif");
    }

    #[test]
    fn take_record_batch() {
        check("take_record_batch");
    }

    #[test]
    fn interleave_record_batch() {
        check("interleave_record_batch");
    }

    #[test]
    fn concat_batches() {
        check("concat_batches");
    }

    #[test]
    fn coalesce() {
        check("coalesce");
    }

    #[test]
    fn law_filter_take() {
        check("law_filter_take");
    }

    #[test]
    fn law_take_take() {
        check("law_take_take");
    }

    #[test]
    fn law_concat_slices() {
        check("law_concat_slices");
    }

    #[test]
    fn law_interleave() {
        check("law_interleave");
    }
}

/// arrow-array and arrow-data: slicing, equality, FFI, ArrayData
mod arrow_array_and_arrow_data {
    use arrow_pbt::registry::check;

    #[test]
    fn slice() {
        check("slice");
    }

    #[test]
    fn law_slice_slice() {
        check("law_slice_slice");
    }

    #[test]
    fn eq() {
        check("eq");
    }

    #[test]
    fn round_trip_ffi() {
        check("round_trip_ffi");
    }

    #[test]
    fn data_ops() {
        check("data_ops");
    }

    #[test]
    fn is_null() {
        check("is_null");
    }

    #[test]
    fn decimal_precision() {
        check("decimal_precision");
    }
}

/// arrow-ipc
mod arrow_ipc {
    use arrow_pbt::registry::check;

    #[test]
    fn round_trip_ipc() {
        check("round_trip_ipc");
    }
}

/// arrow-arith
mod arrow_arith {
    use arrow_pbt::registry::check;

    #[test]
    fn bools() {
        check("bools");
    }

    #[test]
    fn kleene() {
        check("kleene");
    }

    #[test]
    fn arith() {
        check("arith");
    }

    #[test]
    fn neg() {
        check("neg");
    }

    #[test]
    fn arity() {
        check("arity");
    }

    #[test]
    fn bitwise() {
        check("bitwise");
    }

    #[test]
    fn divisor() {
        check("divisor");
    }

    #[test]
    fn aggregate() {
        check("aggregate");
    }

    #[test]
    fn fixed_point() {
        check("fixed_point");
    }

    #[test]
    fn temporal_arith() {
        check("temporal_arith");
    }

    #[test]
    fn interval_scale() {
        check("interval_scale");
    }
}

/// arrow-cast
mod arrow_cast {
    use arrow_pbt::registry::check;

    #[test]
    fn cast() {
        check("cast");
    }

    #[test]
    fn cast_text() {
        check("cast_text");
    }

    #[test]
    fn temporal_cast() {
        check("temporal_cast");
    }

    #[test]
    fn decimal_format() {
        check("decimal_format");
    }
}

/// arrow-string
mod arrow_string {
    use arrow_pbt::registry::check;

    #[test]
    fn length() {
        check("length");
    }

    #[test]
    fn string_ops() {
        check("string_ops");
    }
}

/// arrow-ord and arrow-row
mod arrow_ord_and_arrow_row {
    use arrow_pbt::registry::check;

    #[test]
    fn cmp() {
        check("cmp");
    }

    #[test]
    fn sort() {
        check("sort");
    }

    #[test]
    fn lexsort() {
        check("lexsort");
    }

    #[test]
    fn row_format() {
        check("row_format");
    }

    #[test]
    fn row_encode() {
        check("row_encode");
    }

    #[test]
    fn radix_sort() {
        check("radix_sort");
    }
}
