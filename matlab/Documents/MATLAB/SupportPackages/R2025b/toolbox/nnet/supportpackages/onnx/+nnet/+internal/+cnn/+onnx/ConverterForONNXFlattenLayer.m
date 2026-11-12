classdef ConverterForONNXFlattenLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.FlattenLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForONNXFlattenLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Generate Flatten-->Unsqueeze
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            existingNodeNames   = {nodeProto.name};
            
            % (1) Transpose (permute)
            flattenNodeName         = onnxName;
            flattenNodeName         = makeUniqueName(existingNodeNames, flattenNodeName);
            newNodes(1)             = NodeProto;
            newNodes(1).op_type     = 'Flatten';
            newNodes(1).name     	= flattenNodeName;
            newNodes(1).input       = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            newNodes(1).output      = {flattenNodeName};
            newNodes(1).attribute   = makeAttributeProto('axis', 'INT', 1);
            
            % (3) Unsqueeze(2,3)
            unsqueezeName                   = [onnxName '_Unsqueeze'];
            unsqueezeName                   = makeUniqueName([existingNodeNames, {flattenNodeName}], unsqueezeName);
            unsqueezeInput                  = {flattenNodeName};
            unsqueezeOutput                 = {unsqueezeName};
            [newNodes(2), unsqueezeInit]    = createNodeProto(this, 'Unsqueeze', unsqueezeName, unsqueezeInput, unsqueezeOutput, [2 3]);
            
            nodeProto               = [nodeProto newNodes];
            parameterInitializers   = unsqueezeInit;
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            outputTensorName                  = unsqueezeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = 'nchw';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

