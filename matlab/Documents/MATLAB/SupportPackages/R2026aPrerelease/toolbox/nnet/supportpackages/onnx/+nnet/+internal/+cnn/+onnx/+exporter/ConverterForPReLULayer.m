classdef ConverterForPReLULayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.PReLULayer into ONNX
    
    % Copyright 2024-2025 The MathWorks, Inc.
    methods
        function this = ConverterForPReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [NodeName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            NodeName            = makeUniqueName({nodeProto.name}, NodeName);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            SlopeName         	= [NodeName '_Slope'];
            [newNode, ~] = createNodeProto(this, 'SISO', 'PRelu', NodeName, ...
                mapTensorNames(this, {this.InputLayerNames{1}, SlopeName}, TensorNameMap), {NodeName});
                                   
            % Make parameter Initializer for slope
            t1                  = TensorProto;
            t1.name             = SlopeName;     
            t1.data_type	    = TensorProto_DataType.FLOAT;
            [slope, t1.dims]    = iPermuteSlope(this.NNTLayer.Alpha, inputTensorLayout);
            t1.raw_data         = rawData(single(slope));
            
            nodeProto(end+1)        = newNode;
            parameterInitializers   = t1;
            % Make parameter Inputs
            networkInputs           = [];
            networkOutputs          = [];
                        
            % Update maps
            outputTensorName            = NodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is dangling
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

function [onnxSlope, onnxDims] = iPermuteSlope(DLTSlope, inputTensorLayout)
% Alpha can be a matrix whose shape is unidirectionally broadcastable to
% the size of the input. This function permutes nonscalar Alpha values into
% the equivalent format in ONNX. 
switch inputTensorLayout
    case {'nc'} % BC
        % Alpha has form CB in the DLT layer
        onnxSlope = permute(DLTSlope, [2 1]);
    case {'snc', '1nc'} % TBC or 1BC
        % Alpha has form CBT in the DLT layer
        onnxSlope = permute(DLTSlope, [3 2 1]);
    case {'snch'} % TBCS
        % Alpha has form SCBT in the DLT layer 
        onnxSlope = permute(DLTSlope, [4 3 2 1]);        
    case {'snchw'} % TBCSS
        % Alpha has form SSCBT in the DLT layer 
        onnxSlope = permute(DLTSlope, [5 4 3 1 2]);
    case {'snchwd'} % TBCSSS
        % Alpha has form SSSCBT in the DLT layer        
        onnxSlope = permute(DLTSlope, [6 5 4 1 2 3]);             
    case {'nchw'} % BCSS
        % Alpha has form SSCB in the DLT layer
        onnxSlope = permute(DLTSlope, [4 3 1 2]);
    case {'nchwd'} % BCSSS
        % Alpha has form SSSCB in the DLT layer                
        onnxSlope = permute(DLTSlope, [5 4 1 2 3]);     
    otherwise
        error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));     
end
onnxDims = nnet.internal.cnn.onnx.dimVector(size(onnxSlope), strlength(inputTensorLayout));
end
