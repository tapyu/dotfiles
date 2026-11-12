classdef ConverterForSoftmaxLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a SoftmaxLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForSoftmaxLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            existingNodeNames           = {nodeProto.name};
            [softmaxName, nameChanged]  = legalizeNNTName(this, this.NNTLayer.Name);
            softmaxName                 = makeUniqueName(existingNodeNames, softmaxName);
            DLTInputSize                = this.InputLayerSizes{1};
            inputTensorNames            = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout           = TensorLayoutMap(inputTensorNames{1});
            
            % Determine whether to include transposes, the permutations if
            % so, and the softmax axis:
            if isDLTTensorFlat(this, DLTInputSize)
                includeTransposes = false;
                switch inputTensorLayout
                    case {'nchw', 'nchwd','nc', 'nch'}
                        softmaxAxis = 1;
                    case {'1nc', 'snc','snch', 'snchw', 'snchwd'}
                        softmaxAxis = 2;
                    otherwise
                        error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
                end
            else
                % DLT tensor is not flat
                includeTransposes = true;
                switch inputTensorLayout
                    case 'nch'
                        perm1               = [0 2 1];
                        perm2               = [0 2 1];
                        softmaxAxis         = 2;
                    case 'nchw'
                        perm1               = [0 2 3 1];
                        perm2               = [0 3 1 2];
                        softmaxAxis         = 3;
                    case 'nchwd'
                        perm1               = [0 2 3 4 1];
                        perm2               = [0 4 1 2 3];
                        softmaxAxis         = 4;
                    case 'snch'
                        perm1               = [0 1 3 2];
                        perm2               = [0 1 3 2];
                        softmaxAxis         = 3;
                    case 'snchw'
                        perm1               = [0 1 3 4 2];
                        perm2               = [0 1 4 2 3];
                        softmaxAxis         = 4;
                    case 'snchwd'
                        perm1               = [0 1 3 4 5 2];
                        perm2               = [0 1 5 2 3 4];
                        softmaxAxis         = 5;
                    otherwise
                        error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
                end
            end
            
            % Create the nodes:
            transpose1Node          = [];
            transpose2Node          = [];
            outputTensorName      	= softmaxName;
            if includeTransposes
                % In ONNX, Transpose to put C last:
                transpose1NodeName       = [softmaxName '_Transpose1'];
                transpose1NodeName       = makeUniqueName([existingNodeNames, {softmaxName}], transpose1NodeName);
                [transpose1Node, ~] = createNodeProto(this, 'Transpose', transpose1NodeName, ...
                        mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {transpose1NodeName}, perm1);
                % Transpose back:
                transpose2NodeName       = [softmaxName '_Transpose2'];
                transpose2NodeName       = makeUniqueName([existingNodeNames, {softmaxName, transpose1NodeName}],...
                    transpose2NodeName);
                [transpose2Node, ~] = createNodeProto(this, 'Transpose', transpose2NodeName, ...
                        {softmaxName}, {transpose2NodeName}, perm2);
                outputTensorName         = transpose2NodeName;
                softmaxInputNames        = {transpose1NodeName};
            else
                softmaxInputNames        = inputTensorNames;
            end
            [softmaxNode, ~] = createNodeProto(this, 'Softmax', softmaxName, ...
                        softmaxInputNames, {softmaxName}, softmaxAxis);            
            nodeProto = [nodeProto transpose1Node softmaxNode transpose2Node];
            
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
            
        end
        
        %% override updateNetworkOutputForDanglingLayers. 
        % for 'nchw', 'nchwd' '1nc',  'snchw', 'snchwd' types, if
        % isDLTTensorFlat, insert a flatten operator before softmax layer
        function [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = updateNetworkOutputForDanglingLayers(this, nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap)
            %   In this function, output tensor information was collected from output
            %   tensor's parent node. For each output tensor found, get its DLT size
            %   from the layerAnalyzer's Outputs table, and get its ONNX layout from
            %   the TensorLayoutMap.
     
            if ~this.IsDanglingLayer
                return
            end
            
            import nnet.internal.cnn.onnx.*
            
            
            % Collect information from parent node. Determine the network output
            % tensor size.
            ONNXName                 = this.NNTLayer.Name;
            currentTensorName         = this.mapNodeProtoNametoTensor({ONNXName}, TensorNameMap);
            currentTensorLayout       = TensorLayoutMap(currentTensorName{1});
            dltOutputSize            = this.OutputSize{1};

            
            switch currentTensorLayout
                case {'nch','nchw', 'nchwd'}
                    if isDLTTensorFlat(this, dltOutputSize)
                        % If dltOutputSize is flattened, add a Flatten(1) operator
                        flattenNodeName        = [currentTensorName{1} '_Flatten'];
                        flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                        % Input to be set later
                        FlattenNode            = createNodeProto(this, 'Flatten',...
                            flattenNodeName, {'dummyNode'}, {flattenNodeName}, 1);
                        
                        softmaxNodeName        = currentTensorName{1};           
                        outputTensorName       = softmaxNodeName;
                        nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, FlattenNode, softmaxNodeName, 1);
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize(end))];	% {N C}
                    else
                        % If dltOutputSize is not flattened, create output tensor
                        % directly
                        outputTensorName       = currentTensorName{1};
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {N C H W} or {N C H W D}
                    end
                    
                case 'snc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = currentTensorName{1};
                    outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {1 N C}
                    
                case 'nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = currentTensorName{1};
                    outputTensorSize       = num2cell([this.BatchSizeToExport dltOutputSize]);	% {N C}
                    
                case '1nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    flattenNodeName        = [currentTensorName{1} '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    % Input to be set later
                    FlattenNode            = createNodeProto(this, 'Flatten',...
                        flattenNodeName, {'dummyNode'}, {flattenNodeName}, 2);
                    
                    softmaxNodeName        = currentTensorName{1};
                    outputTensorName       = softmaxNodeName;
                    nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, FlattenNode, softmaxNodeName, 1);
                    outputTensorSize       = num2cell([this.BatchSizeToExport dltOutputSize]);	% {N C}
                    
                case {'snch','snchw', 'snchwd'}
                   
                    if isDLTTensorFlat(this, dltOutputSize)
                        % Insert a node before before existing softmax
                        softmaxNodeName        = currentTensorName{1};
                        
                        shape = [0 0 -1];
                        shapeTensor              = TensorProto;
                        shapeTensor.name         = [currentTensorName{1} '_Reshape_shape'];
                        shapeTensor.name         = makeUniqueName({nodeProto.name}, shapeTensor.name);
                        shapeTensor.data_type    = TensorProto_DataType.INT64;
                        shapeTensor.raw_data     = rawData(int64(shape));
                        shapeTensor.dims         = dimVector(numel(shape),1);
                        
                        reshapeNodeName        = [currentTensorName{1} '_Reshape'];
                        reshapeNodeName        = makeUniqueName({nodeProto.name}, reshapeNodeName);
                        ReshapeNode            = createNodeProto(this, 'SISO', 'Reshape',...
                            reshapeNodeName, {'dummyNode', shapeTensor.name}, {reshapeNodeName});
                        outputTensorName       = softmaxNodeName;
                        outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize(end))];     % {1 N C}
                        nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, ReshapeNode, softmaxNodeName, 1);
                    else
                        % Create output tensor directly
                        outputTensorName        = currentTensorName{1};
                        outputTensorSize        = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {1 N C H W}, or {1 N C H W D}
                    end
                    
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXOutputLayout', this.NNTLayer.Name));
            end
            
            % Set outputs
            netOutputs          = makeValueInfoProtoFromDimensions(outputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
            networkOutputs          = [networkOutputs, netOutputs];
        end
    end
end




