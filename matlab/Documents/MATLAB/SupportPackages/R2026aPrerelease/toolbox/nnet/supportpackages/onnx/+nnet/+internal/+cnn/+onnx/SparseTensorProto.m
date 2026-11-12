classdef SparseTensorProto
    %   optional TensorProto values = 1;
    %   optional TensorProto indices = 2;
    %   repeated int64 dims = 3;
    
    %   Copyright 2019-2021 The MathWorks, Inc.
    
    properties
        values
        indices
        dims
    end
    
    methods
        function this = SparseTensorProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeSparseTensorProto), Ptr);
                [this.values, this.indices, this.dims] = PropertyCell{:};
                 % Call constructors on properties that are Proto objects
                this.values     = arrayfun(@TensorProto, this.values);
                this.indices	= arrayfun(@TensorProto, this.indices);
            end
        end
                
        function encodeSparseTensorProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.values, this.indices, this.dims};
            PtrCell = onnxmex(int32(FuncName.EencodeSparseTensorProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorProto, this.values,       PtrCell{1});
            arrayfun(@encodeTensorProto, this.indices,      PtrCell{2});
        end
    end
end

