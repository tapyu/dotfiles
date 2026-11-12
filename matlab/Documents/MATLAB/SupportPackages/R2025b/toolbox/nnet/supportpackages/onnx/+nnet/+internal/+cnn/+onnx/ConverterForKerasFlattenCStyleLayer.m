classdef ConverterForKerasFlattenCStyleLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.keras.layer.FlattenCStyleLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForKerasFlattenCStyleLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Generate Transpose-->Flatten-->Unsqueeze. The ONNX input
            % tensor is [N C H W]. To match the order of the units in  the
            % flattened MATLAB layer we need [H W C] within each N, or [N H
            % W C]. After that, we flatten with axis=1 to give [N HWC] in
            % row-major. Finally, unsqueeze(2,3) to obtain [N C 1 1].
            import nnet.internal.cnn.onnx.*

            existingNodeNames       = {nodeProto.name};
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
                    
            switch inputTensorLayout
                case 'nchw'
                    % (1) Transpose (permute)
                    TransposeNodeName       = [onnxName '_Transpose'];
                    TransposeNodeName       = makeUniqueName(existingNodeNames, TransposeNodeName);
                    newNodes(1)            = NodeProto;
                    newNodes(1).op_type    = 'Transpose';
                    newNodes(1).name     	= TransposeNodeName;
                    newNodes(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                    newNodes(1).output     = {TransposeNodeName};
                    perm                    = [0 2 3 1];        % Permute [N C H W] into [N H W C]
                    newNodes(1).attribute  = makeAttributeProto('perm', 'INTS', perm);
                    
                    % (2) Flatten
                    FlattenNodeName         = [onnxName '_Flatten'];
                    FlattenNodeName         = makeUniqueName([existingNodeNames, {TransposeNodeName}], FlattenNodeName);
                    newNodes(2)            = NodeProto;
                    newNodes(2).op_type    = 'Flatten';
                    newNodes(2).name       = FlattenNodeName;
                    newNodes(2).input      = {TransposeNodeName};
                    newNodes(2).output     = {FlattenNodeName};
                    axis                    = 1;
                    newNodes(2).attribute  = makeAttributeProto('axis', 'INT', axis);
                    
                    % (3) Unsqueeze(2,3)
                    unsqueezeName                   = [onnxName '_Unsqueeze'];
                    unsqueezeName                   = makeUniqueName([existingNodeNames, {TransposeNodeName, FlattenNodeName}],...
                        unsqueezeName);
                    unsqueezeInput                  = {FlattenNodeName};
                    unsqueezeOutput                 = {unsqueezeName};
                    [newNodes(3), unsqueezeInit]    = createNodeProto(this, 'Unsqueeze', unsqueezeName, unsqueezeInput, unsqueezeOutput, [2 3]);
                    
                    nodeProto               = [nodeProto newNodes];
                    parameterInitializers   = unsqueezeInit;
                    networkInputs           = [];
                    networkOutputs          = [];
                    
                    % Update maps
                    outputTensorName                  = unsqueezeName;
                    TensorNameMap(this.NNTLayer.Name) = outputTensorName;
                    TensorLayoutMap(outputTensorName) = 'nchw';
                    
                case 'snc'
                    % (1) Transpose (permute)
                    TransposeNodeName       = [onnxName '_Transpose'];
                    TransposeNodeName       = makeUniqueName(existingNodeNames, TransposeNodeName);
                    newNodes(1)            = NodeProto;
                    newNodes(1).op_type    = 'Transpose';
                    newNodes(1).name     	= TransposeNodeName;
                    newNodes(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                    newNodes(1).output     = {TransposeNodeName};
                    perm                    = [1 0 2];        % Permute [S N C] into [N S C]
                    newNodes(1).attribute  = makeAttributeProto('perm', 'INTS', perm);
                    
                    % (2) Flatten
                    FlattenNodeName         = [onnxName '_Flatten'];
                    FlattenNodeName         = makeUniqueName([existingNodeNames, {TransposeNodeName}], FlattenNodeName);
                    newNodes(2)            = NodeProto;
                    newNodes(2).op_type    = 'Flatten';
                    newNodes(2).name       = FlattenNodeName;
                    newNodes(2).input      = {TransposeNodeName};
                    newNodes(2).output     = {FlattenNodeName};
                    axis                    = 1;
                    newNodes(2).attribute  = makeAttributeProto('axis', 'INT', axis);
                    
                    nodeProto               = [nodeProto newNodes];
                    parameterInitializers   = [];
                    networkInputs           = [];
                    networkOutputs          = [];
                    
                    % Update maps
                    outputTensorName                  = FlattenNodeName;
                    TensorNameMap(this.NNTLayer.Name) = outputTensorName;
                    TensorLayoutMap(outputTensorName) = 'nc';

                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

