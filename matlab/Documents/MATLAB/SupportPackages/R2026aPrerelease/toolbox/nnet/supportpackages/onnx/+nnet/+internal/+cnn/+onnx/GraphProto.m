classdef GraphProto
    %   repeated NodeProto node = 1;
    %   optional string name = 2;   // namespace Graph
    %   repeated TensorProto initializer = 5;
    %   optional string doc_string = 10;
    %   repeated ValueInfoProto input = 11;
    %   repeated ValueInfoProto output = 12;
    %   repeated ValueInfoProto value_info = 13;
    %   repeated TensorAnnotation quantization_annotation = 14;
    %   repeated SparseTensorProto sparse_initializer = 15;
    
    %   Copyright 2019-2021 The MathWorks, Inc.

    properties
        node
        name
        initializer
        doc_string
        input
        output
        value_info
        quantization_annotation
        sparse_initializer
    end
    
    methods
        function this = GraphProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeGraphProto), Ptr);
                [this.node, this.name, this.initializer, this.doc_string, this.input, ...
                    this.output, this.value_info, this.quantization_annotation, this.sparse_initializer] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                nodeCell            = arrayfun(@NodeProto, this.node, 'UniformOutput',false);
                this.node           = [nodeCell{:}];
                initializerCell     = arrayfun(@TensorProto, this.initializer, 'UniformOutput',false);
                this.initializer	= [initializerCell{:}];
                inputCell           = arrayfun(@ValueInfoProto, this.input, 'UniformOutput',false);
                this.input          = [inputCell{:}];
                this.output         = arrayfun(@ValueInfoProto, this.output);
                this.value_info     = arrayfun(@ValueInfoProto, this.value_info);
                this.quantization_annotation	= arrayfun(@TensorAnnotation, this.quantization_annotation);
                this.sparse_initializer         = arrayfun(@SparseTensorProto, this.sparse_initializer);
            end
        end
        
        function encodeGraphProto(this, CPtr)
            import nnet.internal.cnn.onnx.*
            % Recursively fill the CPtr from 'this'.
            PropertyCell = {this.node, this.name, this.initializer, this.doc_string, this.input, ...
                this.output, this.value_info, this.quantization_annotation, this.sparse_initializer};
            PtrCell = onnxmex(int32(FuncName.EencodeGraphProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeNodeProto, this.node,               PtrCell{1});
            arrayfun(@encodeTensorProto, this.initializer,      PtrCell{3});
            arrayfun(@encodeValueInfoProto, this.input,         PtrCell{5});
            arrayfun(@encodeValueInfoProto, this.output,        PtrCell{6});
            arrayfun(@encodeValueInfoProto, this.value_info,    PtrCell{7});
            arrayfun(@encodeTensorAnnotation, this.quantization_annotation, PtrCell{8});
            arrayfun(@encodeSparseTensorProto, this.sparse_initializer, PtrCell{9});
        end
    end
end
