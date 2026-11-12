classdef TypeProto_SparseTensor
%     optional int32 elem_type = 1;
%     optional TensorShapeProto shape = 2;
    
    %   Copyright 2019-2021 The MathWorks, Inc.

    properties
        elem_type
        shape
    end
    
    methods
        function this = TypeProto_SparseTensor(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto_SparseTensor), Ptr);
                [this.elem_type, this.shape] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.shape)
                    this.shape = TensorShapeProto(this.shape);
                end
            end
        end
        
        function encodeTypeProto_SparseTensor(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.elem_type, this.shape};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto_SparseTensor), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorShapeProto, this.shape, PtrCell{2});
        end
    end
end