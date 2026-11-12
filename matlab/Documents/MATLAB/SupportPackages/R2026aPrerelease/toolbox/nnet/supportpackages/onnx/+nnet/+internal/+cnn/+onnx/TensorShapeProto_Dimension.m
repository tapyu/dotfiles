classdef TensorShapeProto_Dimension
    %     oneof value {
    %       int64 dim_value = 1;
    %       string dim_param = 2;   // namespace Shape
    %     };
    %     optional string denotation = 3;
    
    %   Copyright 2019-2021 The MathWorks, Inc.

    properties
        dim_value
        dim_param
        denotation
    end
    
    methods
        function this = TensorShapeProto_Dimension(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorShapeProto_Dimension), Ptr);
                [this.dim_value, this.dim_param, this.denotation] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                % (none)
            end
        end
        
        function encodeTensorShapeProto_Dimension(this, CPtr)
            % Recursively fill the CPtr from 'this'. To implement 'oneof',
            % any field that is nonempty is writte. So make sure exactly
            % one of them is.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.dim_value, this.dim_param, this.denotation};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorShapeProto_Dimension), CPtr, PropertyCell);
        end
    end
end