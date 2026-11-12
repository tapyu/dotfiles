classdef ConverterForYOLOv2TransformLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a yolov2TransformLayer into ONNX.
    
    % Copyright 2019-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForYOLOv2TransformLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            existingNodeNames            = {nodeProto.name};
            [onnxName, ~]                = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames             = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout            = TensorLayoutMap(inputTensorNames{1});
            
            % (1) Reshape: Takes input X.
            % First create a shape tensor, the second input to the reshape node.
            InputLayerSize               = this.InputLayerSizes{1};
            XH                           = InputLayerSize(1);
            XW                           = InputLayerSize(2);
            XC                           = InputLayerSize(3);
            XA                           = this.NNTLayer.NumAnchorBoxes;
            shape                        = [-1, XC/XA, XA, XH*XW];  
            shapeTensor(1)               = TensorProto;
            shapeTensor(1).name          = [onnxName '_Reshape1_shape'];
            shapeTensor(1).data_type     = TensorProto_DataType.INT64;
            shapeTensor(1).raw_data      = rawData(int64(shape));
            shapeTensor(1).dims          = dimVector(numel(shape),1);
            
            reshapeNodeName              = [onnxName '_Reshape1'];
            reshapeNodeName              = makeUniqueName(existingNodeNames, reshapeNodeName);
            existingNodeNames            = [existingNodeNames, {reshapeNodeName}];
            newNode(1)                   = createNodeProto(this, 'SISO', 'Reshape',...
                reshapeNodeName, {inputTensorNames{1,1}, shapeTensor(1).name}, {reshapeNodeName});
            newParams(1)                 = shapeTensor(1);
            
            parameterInitializers        = newParams;
            
            nodeProto = [nodeProto newNode(1)];
            
            % (2) Split: Takes input from reshape node.
            % Make the nodeProto.
            splitNodeName                = [reshapeNodeName '_Split'];
            splitNodeName                = makeUniqueName(existingNodeNames, splitNodeName);
            existingNodeNames            = [existingNodeNames, {splitNodeName}];
            splitNode1                   = [reshapeNodeName '_Sigmoid_Inp'];
            splitNode2                   = [reshapeNodeName '_Exponential_Inp'];
            splitNode3                   = [reshapeNodeName '_Softmax_Inp']; 
            splitInput                   = {reshapeNodeName};
            splitOutput                  = {splitNode1, splitNode2, splitNode3};

            splitFactor                  = (XC/XA)-5;
            splitAxis                    = 1;
            splits                       = [3, 2, splitFactor];
            [newNode(2), splitInits]     = createNodeProto(this, 'Split', splitNodeName, splitInput, splitOutput, splitAxis, splits);          
            nodeProto                    = [nodeProto newNode(2)];
            parameterInitializers        = [parameterInitializers, splitInits];            
            
            % (3) Sigmoid : Takes input from splitNode1.
            % Make the nodeProto.
            sigmoidNodeName             = [reshapeNodeName '_Sigmoid'];
            sigmoidNodeName             = makeUniqueName(existingNodeNames, sigmoidNodeName);
            existingNodeNames           = [existingNodeNames {sigmoidNodeName}];
            [sigmoidNode, ~]     = createNodeProto(this, 'SISO', 'Sigmoid', sigmoidNodeName, {splitNode1}, {sigmoidNodeName});          

            
            nodeProto = [nodeProto sigmoidNode];
            
            % (3) Exponential : Takes input from splitNode2.
            % Make the nodeProto.
            exponentialNodeName         = [reshapeNodeName '_Exp'];
            exponentialNodeName         = makeUniqueName(existingNodeNames, exponentialNodeName);
            existingNodeNames           = [existingNodeNames, {exponentialNodeName}];
            [exponentialNode, ~]     = createNodeProto(this, 'SISO', 'Exp', exponentialNodeName, {splitNode2}, {exponentialNodeName});          
            nodeProto = [nodeProto exponentialNode];
            
            % (3) Softmax : Takes input from splitNode3.
            % Make the nodeProto.
            perm1            = [0 2 3 1];
            perm2            = [0 3 1 2];
            softmaxAxis      = 3;
            
            % Create the nodes:
            outputTensorName            = [reshapeNodeName '_Softmax'];
            outputTensorName            = makeUniqueName(existingNodeNames, outputTensorName);
            existingNodeNames           = [existingNodeNames, {outputTensorName}];
            
            % In ONNX, Transpose to put C last:
            transpose1NodeName          = [outputTensorName '_Transpose1'];
            transpose1NodeName          = makeUniqueName(existingNodeNames, transpose1NodeName);
            existingNodeNames           = [existingNodeNames, {transpose1NodeName}];
            [transpose1Node, ~] = createNodeProto(this, 'Transpose', transpose1NodeName, {splitNode3}, {transpose1NodeName}, perm1);
            % Transpose back:
            transpose2NodeName          = [outputTensorName '_Transpose2'];
            transpose2NodeName          = makeUniqueName(existingNodeNames, transpose2NodeName);
            existingNodeNames           = [existingNodeNames, {transpose2NodeName}];
            [transpose2Node, ~] = createNodeProto(this, 'Transpose', transpose2NodeName, {outputTensorName}, {transpose2NodeName}, perm2);
            
            softmaxOutputTensorName     = transpose2NodeName;
            softmaxInputNames           = transpose1NodeName;
            [softmaxNode, ~] = createNodeProto(this, 'Softmax', outputTensorName, {softmaxInputNames}, {outputTensorName}, softmaxAxis);
            
            nodeProto = [nodeProto transpose1Node softmaxNode transpose2Node];
            
            % (4) Concat : Takes input from sigmoid, exponential and softmax.
            % Make the nodeProto.
            concatNodeName              = [splitNodeName '_concat'];
            concatNodeName              = makeUniqueName(existingNodeNames, concatNodeName);
            existingNodeNames           = [existingNodeNames, {concatNodeName}]; 
            [concatNode, ~] = createNodeProto(this, 'Concat', concatNodeName, {sigmoidNodeName; exponentialNodeName; softmaxOutputTensorName}, {concatNodeName}, 1);
            nodeProto = [nodeProto concatNode];            
            networkInputs               = [];   

            if ~this.IsDanglingLayer
                % (5) Reshape : Transforms the input back to the original
                % activation size.
                shape2                     = [-1, XC, XH, XW];  
                shape2Tensor               = TensorProto;
                shape2Tensor.name          = [onnxName '_Reshape2_shape'];
                shape2Tensor.data_type     = TensorProto_DataType.INT64;
                shape2Tensor.raw_data      = rawData(int64(shape2));
                shape2Tensor.dims          = dimVector(numel(shape2),1);
                
                reshape2NodeName               = [onnxName '_Reshape2'];
                reshape2NodeName               = makeUniqueName(existingNodeNames, reshape2NodeName);
                reshape2Node                   = createNodeProto(this, 'SISO', 'Reshape',...
                    reshape2NodeName, {concatNodeName, shape2Tensor.name}, {reshape2NodeName});
                parameterInitializers          = [parameterInitializers shape2Tensor];
    
                nodeProto = [nodeProto reshape2Node];            
                
                % Update maps and set network outputs to empty
                TensorNameMap(this.NNTLayer.Name)   = reshape2NodeName;  
                TensorLayoutMap(reshape2NodeName)   = inputTensorLayout;     
                networkOutputs = []; 
            else
                % If the layer is dangling, the output shape will be left
                % as [BatchSize, predictionsPerAnchor,numAnchors, numGrids]
                % after the Concat node.
                
                % Update maps and create TensorProto for network output
                TensorNameMap(this.NNTLayer.Name)   = concatNodeName;  
                TensorLayoutMap(concatNodeName)     = inputTensorLayout;
                outputTensorSize  = [{this.BatchSizeToExport}, {XC/XA, XA, XH*XW}];               
                networkOutputs = makeValueInfoProtoFromDimensions(concatNodeName, TensorProto_DataType.FLOAT, outputTensorSize);                         
            end
        end
    end
end

