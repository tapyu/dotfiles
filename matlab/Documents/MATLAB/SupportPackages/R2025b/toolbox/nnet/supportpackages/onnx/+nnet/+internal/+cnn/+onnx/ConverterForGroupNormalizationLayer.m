classdef ConverterForGroupNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a groupNormalizationLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForGroupNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                =  makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames            = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout           = TensorLayoutMap(inputTensorNames{1});

            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'InstanceNormalization';
            newNode.name      = onnxName;
            newNode.input     = mapTensorNames(this,{this.InputLayerNames{1}, [onnxName '_scale'], [onnxName '_B']}, TensorNameMap);
            newNode.output    = {onnxName};
            newNode.attribute = makeAttributeProto('epsilon', 'FLOAT', this.NNTLayer.Epsilon);
            %if this.OpsetVersion < 7
            %    newNode.attribute(1,2) = makeAttributeProto('is_test', 'INT', 1);
            %end

            % Make parameter Initializers for: scale, B, mean, var
            t1 = TensorProto;
            t1.name = [onnxName '_scale'];
            t1.data_type = TensorProto_DataType.FLOAT;
            t1.raw_data = rawData(single(this.NNTLayer.Scale));
            t1.dims = dimVector(numel(this.NNTLayer.Scale),1);           % NNT data: 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(this.NNTLayer.Offset));
            t2.dims = dimVector(numel(this.NNTLayer.Offset),1);          % NNT data: 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

            parameterInitializers = [t1 t2];

            nodeProto(end+1)        = newNode;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

