classdef TranslatorForPRelu < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX PRelu operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if supportedFormat(this, inputTensorFormats(1), outputTensorFormats(1))
                % slope is input 2
                if numel(this.Node.input)~=2
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:PReluNumargs'))];
                    return;
                elseif ~isTensorInitializer(this.GraphProtoManager, this.Node.input{2})
                    % Slope is not an Initializer
                    return;
                end
                
                % Import slope and create layer. If the user is importing
                % into a dlnework, create preluLayer (which is only
                % supported in dlnetwork). If the user has requested a
                % dagnetwork, then create the legacy nnet.onnx.layer.PreluLayer.
                if this.TargetNetwork=="dlnetwork"
                    [Layer, issues] = translateLayerForDlnetwork(this, inputTensorFormats);
                else % dagnetwork
                    [Layer, issues] = translateLayerForDAG(this, inputTensorFormats);                    
                end
            end         
        end
    end
    
    methods(Access=private)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCSS", "BCSSS", "BC", "BCT", "TBC", "1BC", "T1BC", "TBCSS", "TBCSSS"];
            if ismember(inputTensorFormat, supportedFormats) && isequal(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end

        function [Layer, issues] = translateLayerForDAG(this, inputTensorFormats) 
            % Parse the ONNX Prelu node to construct an
            % nnet.onnx.layer.PreluLayer, which is supported in DAGNetwork
            slopeName	= this.Node.input{2};            
            slope    	= double((initializerRawData(this.GraphProtoManager, slopeName)));
            slopeDim 	= double((initializerSize(this.GraphProtoManager, slopeName)));            
            numChannels = numel(slope);
            if ~supportedSlope(this, inputTensorFormats(1), slopeDim)
                issues = [nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:PReluSlopeDim'))];
                Layer = [];
            else
                DLTChannelDim = getDLTChannelDim(this, inputTensorFormats(1));
                [Layer, issues] = constructLayer(this, 'nnet.onnx.layer.PreluLayer', this.Node.name, this.Node, this.Node.name, numChannels, DLTChannelDim, slope);
            end      
        end
        
        function [Layer, issues] = translateLayerForDlnetwork(this, inputTensorFormats)
            % Parse the ONNX Prelu node to construct a DLT preluLayer, which is 
            % supported only in dlnetwork.
            slopeName	= this.Node.input{2};
            slopeDim 	= double((initializerSize(this.GraphProtoManager, slopeName)));
            rawSlope    = double((initializerRawData(this.GraphProtoManager, slopeName)));
            slope       = reshape(rawSlope, [slopeDim ones(1, 2-numel(slopeDim))]);
            DLTSlope    = iPermuteONNXSlopeToDLT(slope, slopeDim, inputTensorFormats(1));  
            layerName   = this.Node.name;
            [Layer, issues] = constructLayer(this, 'preluLayer', layerName, this.Node, "Name", layerName, "Alpha", DLTSlope);            
        end        
    end
end

function alpha = iPermuteONNXSlopeToDLT(onnxSlope, onnxSlopeShape, inputTensorFormat)
% Permute the slope to the equivalent Alpha in DLT
% inputTensorFormat indicates the label of the incoming tensor, not of
% the slope. The slope can be any shape that is broadcastable to the input
% format. For example, if inputTensorFormat is 'TBCS', the slope could be
% any one of TBCS, BCS, CS, S, or a scalar. 
% To convert slope into the equivalent Alpha in preluLayer, we reshape
% the slope so that it has the same dimensionality as the
% inputTensorFormat, and then apply a permutation vector to convert it into
% DLT format. For example,
% if inputTensorFormat = BCSS and slope has shape CSS:
% 1. paddedSlope = reshape(slope, [1 Csize S1size S2size]);  % CSS -> 1CSS
% 2. alpha = permute(paddedSlope, [3 4 2 1]);                % 1CSS -> SSC

if isscalar(onnxSlope)
    alpha = onnxSlope;
else
    reshapedSlope   = iReshapeSlope(onnxSlope, onnxSlopeShape, inputTensorFormat);
    perm            = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.permONNXToDLT(inputTensorFormat);
    alpha           = permute(reshapedSlope, perm);   
end
end

function reshapedSlope = iReshapeSlope(onnxSlope, onnxSlopeShape, inputTensorFormat)
% Reshape onnxSlope by adding leading singleton dimensions up to the length
% of inputTensorFormat
nDimsFormat = strlength(inputTensorFormat);
numLeadingSingletonDims = nDimsFormat - numel(onnxSlopeShape);
if numLeadingSingletonDims > 0
    reshapedSlope = reshape(onnxSlope, ...
        [ones(1, numLeadingSingletonDims), onnxSlopeShape]);
else
    reshapedSlope = onnxSlope;
end
end

function tf = supportedSlope(~, inputTensorFormat, slopeDim)
    % Here we do no ONNX legality checking, but assume it is legal
    % in ONNX. We just check for the cases that nnet.onnx.layer.PreluLayer
    % supports, which is either a scalar slope, or a vector slope where the only
    % non-singleton dimension is in the C location.
    if ismember(inputTensorFormat, ["BC", "TBC", "1BC", "T1BC"])
        % Check for singletons in all but the last dimension:
        tf = all(slopeDim(1:numel(slopeDim)-1)==1);                             % slopeDim is [1* C]
    elseif inputTensorFormat=="BCT"
        % Broadcast to length 3 then check for singletons in dims 1 and 3:
        fullSlopeDim = [ones(1,3-numel(slopeDim)) slopeDim];
        tf = isequal(fullSlopeDim([1 3]), [1 1]);
    elseif inputTensorFormat=="BCSS"
        % Broadcast to length 4 then check for singletons in dims 1,3,4:
        fullSlopeDim = [ones(1,4-numel(slopeDim)) slopeDim];
        tf = isequal(fullSlopeDim([1 3 4]), [1 1 1]);
    elseif inputTensorFormat=="TBCSS"
        % Broadcast to length 5, then check for singletons in dims 1,2,4,5:
        fullSlopeDim = [ones(1,5-numel(slopeDim)) slopeDim];
        tf = isequal(fullSlopeDim([1 2 4 5]), [1 1 1 1]);
    elseif inputTensorFormat=="TBCSSS"
        % Broadcast to length 6, then check for singletons in dims 1,2,4,5,6:
        fullSlopeDim = [ones(1,6-numel(slopeDim)) slopeDim];
        tf = isequal(fullSlopeDim([1 2 4 5 6]), [1 1 1 1 1]);        
    else % BCSSS
        % Broadcast to length 5 then check for singletons in dims 1,3,4,5:
        fullSlopeDim = [ones(1,5-numel(slopeDim)) slopeDim];
        tf = isequal(fullSlopeDim([1 3 4 5]), [1 1 1 1]);
    end
end

function DLTChannelDim = getDLTChannelDim(~, inputTensorFormat)
    if ismember(inputTensorFormat, ["BC", "BCT", "TBC", "1BC", "T1BC"])
        DLTChannelDim = 1;                                          % CB, CBT
    elseif ismember(inputTensorFormat, ["BCSS", "TBCSS"])
        DLTChannelDim = 3;                                          % SSCB, SSCBT
    else % BCSSS, TBCSSS
        DLTChannelDim = 4;                                          % SSSCB, SSSCBT
    end
end