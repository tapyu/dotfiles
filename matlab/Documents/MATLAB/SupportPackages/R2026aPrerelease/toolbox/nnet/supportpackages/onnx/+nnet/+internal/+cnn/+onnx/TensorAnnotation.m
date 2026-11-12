classdef TensorAnnotation
    % Based on the protobuf ONNX message:
    % message TensorAnnotation {
    %   optional string tensor_name = 1;
    %   repeated StringStringEntryProto quant_parameter_tensor_names = 2;
    % }
    
    %   Copyright 2019-2021 The MathWorks, Inc.

    properties
        tensor_name
        quant_parameter_tensor_names
    end
    
    methods
        function this = TensorAnnotation(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get shallow properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorAnnotation), Ptr);
                [this.tensor_name, this.quant_parameter_tensor_names] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                this.quant_parameter_tensor_names = arrayfun(@StringStringEntryProto, this.quant_parameter_tensor_names);
            end
        end
    end
    
    methods
        function encodeTensorAnnotation(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            % Fill the TensorAnnotation shallowly. Allocate sub-objects, returning pointers.
            PropertyCell = {this.tensor_name, this.quant_parameter_tensor_names};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorAnnotation), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeStringStringEntryProto, this.quant_parameter_tensor_names, PtrCell{2});
        end
    end
end

