classdef TypeProto
    %   oneof value {
    %     Tensor tensor_type = 1;
    %     Sequence sequence_type = 4;
    %     Map map_type = 5;
    %     Opaque opaque_type = 7;
    %     SparseTensor sparse_tensor_type = 8;
    %     Optional optional_type = 9;
    %   }
    %   optional string denotation = 6;
    
    %   Copyright 2019-2024 The MathWorks, Inc.
    properties
        tensor_type
        sequence_type
        map_type
        opaque_type
        sparse_tensor_type
        optional_type
        denotation
    end
    
    methods
        function this = TypeProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto), Ptr);
                [this.tensor_type, this.sequence_type, this.map_type, this.opaque_type,...
                    this.sparse_tensor_type, this.optional_type, this.denotation] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.tensor_type)
                    this.tensor_type = TypeProto_Tensor(this.tensor_type);
                end
                if ~isempty(this.sequence_type)
                    this.sequence_type = TypeProto_Sequence(this.sequence_type);
                end
                if ~isempty(this.map_type)
                    this.map_type = TypeProto_Map(this.map_type);
                end
                if ~isempty(this.opaque_type)
                    this.opaque_type = TypeProto_Opaque(this.opaque_type);
                end
                if ~isempty(this.sparse_tensor_type)
                    this.sparse_tensor_type = TypeProto_SparseTensor(this.sparse_tensor_type);
                end
                if ~isempty(this.optional_type)
                    this.optional_type = TypeProto_Optional(this.optional_type);
                end
            end
        end
        
        function encodeTypeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.tensor_type, this.sequence_type, this.map_type, ...
                this.opaque_type, this.sparse_tensor_type, this.optional_type, this.denotation};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTypeProto_Tensor, this.tensor_type,     PtrCell{1});
            arrayfun(@encodeTypeProto_Sequence, this.sequence_type, PtrCell{2});
            arrayfun(@encodeTypeProto_Map, this.map_type,           PtrCell{3});
            arrayfun(@encodeTypeProto_Opaque, this.opaque_type,     PtrCell{4});
            arrayfun(@encodeTypeProto_SparseTensor, this.sparse_tensor_type, PtrCell{5});
            arrayfun(@encodeTypeProto_Optional, this.optional_type, PtrCell{6});
        end
    end
end
