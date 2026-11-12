classdef ConverterForAdaptiveAveragePooling2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an adaptiveAveragePooling2dLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForAdaptiveAveragePooling2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*           

            nntLayer                = this.NNTLayer;
            nntLayerName            = nntLayer.Name;
            [onnxName, ~]           = legalizeNNTName(this, nntLayerName);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);

            % Adaptive average pooling does not have a direct equivalent in ONNX,
            % so convert it into an average pooling operator and issue a warning that
            % the results may differ for certain output sizes. Global
            % average pooling (OutputSize=[1,1]) is the exception to the
            % rule.
            if ~isequal(nntLayer.OutputSize, [1,1])
                warning(message('nnet_cnn_onnx:onnx:AdaptivePoolingResultsMayDiffer', this.NNTLayer.Name));
            end

            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            outputTensorLayout = inputTensorLayout;            

            % Verify that the input tensor format is supported
            iAssertSupportedInputTensorFormat(inputTensorLayout, this.NNTLayer.Name);

            % If the input tensor format is temporal, insert a transpose
            % operator before the average pooling operator
            nextOperatorInput = this.InputLayerNames{1};
            %  AvgPool to operate over TBCS -> BCST
            if isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 0]);
            %  AvgPool to operate over TBCSS -> BCSST
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 0]);
            end

            % Determine the appropriate pooling size, stride, and
            % padding values based on the adaptiveAveragePooling2dLayer's
            % output size.
            % Because these parameters are attributes of the
            % AveragePool node rather than inputs, they are hard-coded in
            % the operator and can not be determined dynamically based on
            % the operator input size. This network will therefore only
            % work for the input size determined by the network analyzer
            % during export (as stored in this.InputLayerSizes).
            dltInputSize = this.FullInputSizes{1};
            inputSize = iGetInputPoolingSize(dltInputSize, inputTensorLayout, nntLayerName);            
            [poolSize, stride, paddingSize] = iTransformAveragePoolingParameters(inputSize, nntLayer.OutputSize, inputTensorLayout);
            
            % Create the AveragePool node
            avgPoolNode           = NodeProto;
            avgPoolNode.op_type   = 'AveragePool';
            avgPoolNode.name      = onnxName;
            avgPoolNode.input     = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            avgPoolNode.output    = {onnxName};
            avgPoolNode.attribute = [...
                makeAttributeProto('kernel_shape', 'INTS', poolSize),...
                makeAttributeProto('pads',         'INTS', paddingSize),...
                makeAttributeProto('strides',      'INTS', stride)
                ];
            
            nextOperatorInput       = onnxName;
            nodeProto(end+1)        = avgPoolNode;

            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension -> BCST to TBCS
            if isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [3 0 1 2]);
            %the sequence dimension -> BCSST to TBCSS
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [4 0 1 2 3]);
            end

            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name)   = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput)  = outputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

function iAssertSupportedInputTensorFormat(inputTensorLayout, layerName)
if ~ismember(inputTensorLayout, ["nchw", "snch", "snchw"]) % BCSS, TBCS, and TBCSS respectively
    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', layerName));
end
if inputTensorLayout=="snch"
    warning(message('nnet_cnn_onnx:onnx:AdaptivePoolParamsMinLength'));
end
end

function [poolSize, stride, paddingSize] = iTransformAveragePoolingParameters(inputSize, outputSize, inputTensorLayout)
% Calculate the kernel, stride, and padding values for regular average
% pooling that would produce the same output size, using the formula:
% stride = floor(inputSize, outputSize)
% kernel = inputSize - (outputSize - 1) .* stride
% inputSize and outputSize should both be two-element vectors containing
% the sizes of the dimensions that are being pooled.

stride = double(idivide(int64(inputSize),int64(outputSize),'floor')); 
poolSize = min(inputSize - (outputSize - 1) .* stride, inputSize); % min ensures the poolSize is not made larger than the input
paddingSize = [0 0 0 0];

switch inputTensorLayout
    case "snchw" % TBCSS
        poolSize           = [poolSize 1]; %Kernels of size 1 indicate no-ops over sequence dimension    
        stride             = [stride 1];
        paddingSize        = [paddingSize 0 0];
end
end

function inputSize = iGetInputPoolingSize(dltInputSize, inputTensorLayout, layerName)
% Pad dltInputSize up to the number of dimensions in inputTensorLayout
maxNd = max(strlength(inputTensorLayout), numel(dltInputSize));
dltInputSize = [dltInputSize ones(1, maxNd - numel(dltInputSize))];
switch inputTensorLayout
    case {'nchw', 'snchw'} % SSCB or SSCBT
        inputSize = dltInputSize(1:2);
    case 'snch' % SCBT
        inputSize = [dltInputSize(1) dltInputSize(4)];  
    otherwise
        error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', layerName));
end
end
