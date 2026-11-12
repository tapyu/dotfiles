classdef ConverterForUnsupportedOutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an Unsupported Output Layer into ONNX. If there are
    % no other converters to convert output layers, we generically produce
    % an output tensor. 

    % Copyright 2021 The MathWorks, Inc.

    methods
        function this = ConverterForUnsupportedOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorNames        = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            parameterInitializers   = [];
            networkInputs           = [];

            % We use the ONNX Tensor layout to get the number of dimensions
            % of the ONNX output tensor. 
            outputParams     = cell(1, numel(TensorLayoutMap(inputTensorNames{1}))); 
            outputParams(:)  = {[inputTensorNames{1} '_dim']};
            % All output shapes are marked as variable/unknown. 
            outputParams = matlab.lang.makeUniqueStrings(outputParams); 
            networkOutputs = makeValueInfoProtoFromDimensions(inputTensorNames{1}, TensorProto_DataType.FLOAT, outputParams);
        end
    end
end