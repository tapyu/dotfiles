classdef ConverterForMaxUnpooling2DLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a MaxUnpooling2dLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForMaxUnpooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            if this.OpsetVersion <= 8
                warning(message('nnet_cnn_onnx:onnx:MaxUnpoolOpset9Preferred', this.NNTLayer.Name));
                newNode           = NodeProto;
                newNode.op_type   = 'MaxUnpool';
                newNode.domain    = 'com.mathworks';
                newNode.name      = onnxName;
                newNode.input     = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
                newNode.output    = {onnxName};
                newNode.doc_string = 'maxUnpooling2dLayer';
            else
                newNode           = NodeProto;
                newNode.op_type   = 'MaxUnpool';
                newNode.name = onnxName;
                newNode.input = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
                newNode.output = {onnxName};

                % lookup the maxpool node proto which is the node that
                % produces the second input tensor to this layer.
                maxPoolTensorname = mapTensorNames(this, this.InputLayerNames(2), TensorNameMap);
                maxpoolParentNodeIdx = arrayfun(@(node)(ismember(maxPoolTensorname, node.output)), nodeProto);
                maxpoolParentNodeAttr = nodeProto(maxpoolParentNodeIdx).attribute;

                % lookup the stride attribute of the parent MaxUnpool
                % operator.
                stride = maxpoolParentNodeAttr(strcmp({maxpoolParentNodeAttr.name}, 'strides')).ints;

                % We set the unpool kernel_shape = strides since in DLT,
                % maxPooling2dLayer requires PoolSize <= Stride
                newNode.attribute = [...
                    makeAttributeProto('kernel_shape',  'INTS', stride)...
                    makeAttributeProto('strides', 'INTS', stride)];
            end

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName = onnxName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
