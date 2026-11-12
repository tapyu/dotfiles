classdef ConverterForSwishLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a SwishLayer into ONNX
        
    % Copyright 2020-2025 The MathWorks, Inc.

    methods
        function this = ConverterForSwishLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers,...
                networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap]...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % An NNT SwishLayer translates into a Sigmoid followed by an 
            % element-wise multiplication.
            import nnet.internal.cnn.onnx.*
            
            existingNodeNames = {nodeProto.name};
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            
            % (1) Sigmoid: Takes input X
            SigmoidNodeName        = [ONNXName '_Sigmoid'];
            SigmoidNodeName        = makeUniqueName(existingNodeNames, SigmoidNodeName);
            newNodes(1)                 = createNodeProto(this, 'SISO', 'Sigmoid',...
                SigmoidNodeName,mapTensorNames(this, this.InputLayerNames(1), TensorNameMap),{SigmoidNodeName});

            % (2) Multiplication: Takes input SigmoidNodeName and the
            % original X
            MultiplicationNodeName     = [ONNXName '_Mul'];
            MultiplicationNodeName     = makeUniqueName([existingNodeNames {SigmoidNodeName}], MultiplicationNodeName);
            newNodes(2)                 = createNodeProto(this, 'SISO', 'Mul',...
                MultiplicationNodeName, mapTensorNames(this, {SigmoidNodeName, this.InputLayerNames{1}}, TensorNameMap),...
                {MultiplicationNodeName});

            nodeProto               = [nodeProto newNodes];
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            outputTensorName                  = MultiplicationNodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end