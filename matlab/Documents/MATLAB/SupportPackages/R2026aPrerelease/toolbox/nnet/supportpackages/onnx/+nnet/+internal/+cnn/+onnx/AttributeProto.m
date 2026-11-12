classdef AttributeProto
    %   optional string name = 1;           // namespace Attribute
    %   optional string doc_string = 13;
    %   optional AttributeType type = 20;   // discriminator that indicates which field below is in use
    %   optional float f = 2;               // float
    %   optional int64 i = 3;               // int
    %   optional bytes s = 4;               // UTF-8 string
    %   optional TensorProto t = 5;         // tensor value
    %   optional GraphProto g = 6;          // graph
    %   optional SparseTensorProto sparse_tensor = 22;  // sparse tensor value
    %   optional TypeProto tp = 14;         // type proto
    %   repeated float floats = 7;          // list of floats
    %   repeated int64 ints = 8;            // list of ints
    %   repeated bytes strings = 9;         // list of UTF-8 strings
    %   repeated TensorProto tensors = 10;  // list of tensors
    %   repeated GraphProto graphs = 11;    // list of graph
    %   repeated SparseTensorProto sparse_tensors = 23; // list of sparse tensors%
    %   repeated TypeProto type_protos = 15;// list of type protos
    %   optional string ref_attr_name = 21;
    
    %   Copyright 2019-2024 The MathWorks, Inc.
    properties
        name
        doc_string
        type
        f
        i
        s
        t
        g
        sparse_tensor
        tp
        floats
        ints
        strings
        tensors
        graphs
        sparse_tensors
        type_protos
        ref_attr_name
    end
    
    methods
        function this = AttributeProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeAttributeProto), Ptr);
                [this.name, this.doc_string, this.type, this.f, this.i, ...
                    this.s, this.t, this.g, this.sparse_tensor, this.tp, this.floats, this.ints, this.strings, ...
                    this.tensors, this.graphs, this.sparse_tensors, this.type_protos, this.ref_attr_name] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.type)
                    this.type = AttributeProto_AttributeType(this.type);
                end
                if ~isempty(this.t)
                    this.t = TensorProto(this.t);
                end
                if ~isempty(this.g)
                    this.g = GraphProto(this.g);
                end
                if ~isempty(this.sparse_tensor)
                    this.sparse_tensor = SparseTensorProto(this.sparse_tensor);
                end
                if ~isempty(this.tp)
                    this.tp = TypeProto(this.tp);
                end

                this.tensors = arrayfun(@TensorProto, this.tensors);
                this.graphs  = arrayfun(@GraphProto, this.graphs);
                this.sparse_tensors = arrayfun(@SparseTensorProto, this.sparse_tensors);
                this.type_protos = arrayfun(@TypeProto, this.type_protos);
            end
        end
        
        function encodeAttributeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.name, this.doc_string, int32(this.type), this.f, this.i, ...
                this.s, this.t, this.g, this.sparse_tensor, this.tp, this.floats, this.ints, this.strings, ...
                this.tensors, this.graphs, this.sparse_tensors, this.type_protos, this.ref_attr_name};
            PtrCell = onnxmex(int32(FuncName.EencodeAttributeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorProto, this.t, PtrCell{7});
            arrayfun(@encodeGraphProto, this.g, PtrCell{8});
            arrayfun(@encodeSparseTensorProto, this.sparse_tensor, PtrCell{9});
            arrayfun(@encodeTypeProto, this.tp, PtrCell{10});
            arrayfun(@encodeTensorProto, this.tensors, PtrCell{14});
            arrayfun(@encodeGraphProto, this.graphs, PtrCell{15});
            arrayfun(@encodeSparseTensorProto, this.sparse_tensors, PtrCell{16});
            arrayfun(@encodeTypeProto, this.type_protos, PtrCell{17});
        end
    end
end