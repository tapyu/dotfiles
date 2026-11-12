classdef ConverterForFlattenLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.FlattenLayer into ONNX
    
    % Copyright 2019-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForFlattenLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Generate Transpose->Reshape, Reshape or no-op
            import nnet.internal.cnn.onnx.*
            [onnxName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});            
            
            switch inputTensorLayout
                % No-op cases
                case {'snc', '1nc', 'nc'}
                    newNodes = [];     
                    newParams = [];
                    outputTensorName = inputTensorNames{1}; % No new node is added
                
                % Reshape cases
                case {'nch', 'snch'}
                    % Reshape
                    % Assign a shape tensor, which is the second input to the
                    % reshape node
                    if strcmp(inputTensorLayout, 'nch')
                        shape = [0 -1]; % NCH -> NC
                    else % SNCH -> SNC
                        shape = [0 0 -1];
                    end

                    shapeTensor              = TensorProto;
                    shapeTensor.name         = [onnxName '_Reshape_shape'];
                    shapeTensor.data_type    = TensorProto_DataType.INT64;
                    shapeTensor.raw_data     = rawData(int64(shape));
                    shapeTensor.dims         = dimVector(numel(shape),1);
                    
                    existingNodeNames        = {nodeProto.name};
                    reshapeNodeName          = [onnxName '_Reshape'];
                    reshapeNodeName          = makeUniqueName(existingNodeNames, reshapeNodeName);
                    newNodes(1)              = NodeProto;
                    newNodes(1).op_type      = 'Reshape';
                    newNodes(1).name         = reshapeNodeName;
                    newNodes(1).input        = {inputTensorNames{1}, shapeTensor.name};                          
                    newNodes(1).output       = {reshapeNodeName};
                    newParams                = shapeTensor; 
                    outputTensorName         = reshapeNodeName;

                % Transpose -> Reshape cases
                case {'snchw', 'snchwd', 'nchw', 'nchwd'}                   
                    % To flatten in row-major order in the generated ONNX node,
                    % we need to permute the input. Assign a shape tensor
                    % based on flattened output shape
                    if strcmp(inputTensorLayout, 'snchw')
                        perm = [0 1 2 4 3]; % SNCHW -> SNCWH
                        shape = [0 0 -1]; % SNCWH -> SNC
                    elseif strcmp(inputTensorLayout, 'snchwd')
                        perm = [0 1 2 5 4 3]; % SNCHWD -> SNCDWH
                        shape = [0 0 -1]; % SNDWH -> SNC
                    elseif strcmp(inputTensorLayout, 'nchw')
                        perm = [0 1 3 2]; % NCHW -> NCWH
                        shape = [0 -1]; % NCWH -> NC
                    elseif strcmp(inputTensorLayout, 'nchwd')
                        perm = [0 1 4 3 2]; % NCHWD -> NCDWH
                        shape = [0 -1]; %NCDWH -> NC
                    end
                    
                    % (1) Transpose (permute)
                    existingNodeNames        = {nodeProto.name};
                    TransposeNodeName        = [onnxName '_Transpose'];
                    TransposeNodeName        = makeUniqueName(existingNodeNames, TransposeNodeName);
                    newNodes(1)              = NodeProto;
                    newNodes(1).op_type      = 'Transpose';
                    newNodes(1).name         = TransposeNodeName;
                    newNodes(1).input        = inputTensorNames(1);
                    newNodes(1).output       = {TransposeNodeName};
                    newNodes(1).attribute    = makeAttributeProto('perm', 'INTS', perm);
                    
                    % (2) Reshape
                    shapeTensor              = TensorProto;
                    shapeTensor.name         = [onnxName '_Reshape_shape'];
                    shapeTensor.data_type    = TensorProto_DataType.INT64;
                    shapeTensor.raw_data     = rawData(int64(shape));
                    shapeTensor.dims         = dimVector(numel(shape),1);
                    
                    reshapeNodeName          = [onnxName '_Reshape'];
                    reshapeNodeName          = makeUniqueName([existingNodeNames, {TransposeNodeName}],...
                        reshapeNodeName);
                    newNodes(2)              = NodeProto;
                    newNodes(2).op_type      = 'Reshape';
                    newNodes(2).name         = reshapeNodeName;
                    newNodes(2).input        = {TransposeNodeName, shapeTensor.name};                          
                    newNodes(2).output       = {reshapeNodeName};
                    newParams                = shapeTensor; 
                    outputTensorName         = reshapeNodeName;

                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));                    
            end
            
            nodeProto               = [nodeProto newNodes];
            parameterInitializers   = newParams;
            networkInputs           = [];
            networkOutputs          = [];
            
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            switch inputTensorLayout
                case {'nc', 'nch', 'nchw', 'nchwd'}
                    TensorLayoutMap(outputTensorName) = 'nc';
                case {'snc' 'snchw','snchwd','snch'}
                    TensorLayoutMap(outputTensorName) = 'snc';
                case '1nc'
                    TensorLayoutMap(outputTensorName) = '1nc';
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
