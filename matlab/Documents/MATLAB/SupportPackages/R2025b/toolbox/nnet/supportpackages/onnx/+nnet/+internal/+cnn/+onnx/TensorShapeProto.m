classdef TensorShapeProto
    %   repeated Dimension dim = 1;

    %   Copyright 2018-2021 The MathWorks, Inc.    

    properties
        dim
    end
    
    methods
        function this = TensorShapeProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorShapeProto), Ptr);
                [this.dim] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                this.dim = arrayfun(@TensorShapeProto_Dimension, this.dim);
            end
        end
        
        function encodeTensorShapeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.dim};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorShapeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorShapeProto_Dimension, this.dim, PtrCell{1});
        end
    end
end