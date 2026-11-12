classdef ConverterForSpatialDropoutLayer < nnet.internal.cnn.onnx.ConverterForDropoutLayer
    % Class to convert a spatialDropoutLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForSpatialDropoutLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForDropoutLayer(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            % ONNX does not have a spatial dropout operator, so export the
            % layer to a regular Dropout operator instead, and throw a
            % warning. 
            % At inference time there is no difference, but the training
            % behavior will differ.
            warning(message('nnet_cnn_onnx:onnx:SpatialDropoutResultsMayDiffer', this.NNTLayer.Name)); 
            [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx@nnet.internal.cnn.onnx.ConverterForDropoutLayer(this, nodeProto, TensorNameMap, TensorLayoutMap);
        end
    end
end

