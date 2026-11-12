classdef ConverterForCustomLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert an unknown custom layer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    properties
        LayerAnalyzer
    end

    methods
        function this = ConverterForCustomLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % An NNT unknown custom layer translates into a com.mathworks.Custom operator.
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Input names are the input tensors and learnable parameters of
            % the layer
            LearnableNames          = this.LayerAnalyzer.Learnables.Properties.RowNames(:)';
            LearnableONNXNames      = cellfun(@(name)legalizeNNTName(this, name), LearnableNames, 'UniformOutput', false);
            AllInputNames           = mapTensorNames(this, [this.InputLayerNames(:)', LearnableONNXNames], TensorNameMap);

            % Output names are the output tensors of the layer. If there is
            % only one output, change it to the layer name
            OutputNames             = this.LayerAnalyzer.Outputs.Properties.RowNames(:)';
            if numel(OutputNames)==1
                OutputNames         = {onnxName};
            end
            OutputONNXNames         = cellfun(@(name)legalizeNNTName(this, name), OutputNames, 'UniformOutput', false);

            % Make the attributes
            HyperNames              = this.LayerAnalyzer.Hypers.Properties.RowNames(:)';
            HyperONNXNames          = cellfun(@(name)legalizeNNTName(this, name), HyperNames, 'UniformOutput', false);
            HyperValues             = this.LayerAnalyzer.Hypers.Value(:)';
            Attributes              = cellfun(@attributeFromHyper, HyperONNXNames, HyperValues);
            if isempty(Attributes)
                Attributes = [];
            end

            % Make the nodeProto
            newNode               = NodeProto;
            newNode.op_type       = 'Custom';
            newNode.domain        = 'com.mathworks';
            newNode.name          = onnxName;
            newNode.input         = AllInputNames;
            newNode.output        = OutputONNXNames;
            newNode.doc_string	= getString(message('nnet_cnn_onnx:onnx:CustomLayerDocString'));
            newNode.attribute     = Attributes;

            nodeProto(end+1)        = newNode;

            % Make Initializers for learnable parameters
            parameterInitializers   = cellfun(@(OrigParamName, ONNXParamName)makeInitializer(this.NNTLayer, OrigParamName, ONNXParamName),...
                LearnableNames, LearnableONNXNames);
            networkInputs           = [];
            networkOutputs          = [];

            TensorNameMap(this.NNTLayer.Name) = onnxName;
            % Add output names to layerMap
            for i=1:numel(OutputNames)
                TensorNameMap(OutputNames{i}) = OutputONNXNames{i};
            end
            TensorLayoutMap(outputTensorName) = inputTensorLayout;    % Assume layout is unchanged.

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

function initializer = makeInitializer(NNTLayer, OrigParamName, ONNXParamName)
import nnet.internal.cnn.onnx.*
ParamValue                  = NNTLayer.(OrigParamName);
initializer                 = TensorProto;
initializer.name            = ONNXParamName;
initializer.dims            = dimVector(size(ParamValue), numel(size(ParamValue)));
if isnumeric(ParamValue)
    initializer.data_type	= TensorProto_DataType.FLOAT;
    initializer.raw_data    = rawData(single(ParamValue));
else
    initializer.data_type	= TensorProto_DataType.STRING;
    initializer.raw_data    = rawData(char(ParamValue));
end
end

function attribute = attributeFromHyper(HyperONNXName, HyperValue)
import nnet.internal.cnn.onnx.*
if isnumeric(HyperValue)
    attribute = makeAttributeProto(HyperONNXName, 'FLOATS', HyperValue);
else
    attribute = makeAttributeProto(HyperONNXName, 'STRING', char(HyperValue));
end
end
