classdef NodeTranslation
    % A class to translate a NodeProto into functional code.
    
    %   Copyright 2020-2025 The MathWorks, Inc.    
    properties
        % A struct of state properties with their initial values introduced
        % in this node and its subgraphs. (Currently, this is only
        % batchnorm running means and variances). States introduced in this
        % node will have empty initial values until assigned while parsing
        % the enclosing graph.
        InitialState = struct;
        
        % A struct of nonlearnable parameters with their values that are
        % introduced in this node and its subgraphs.
        Nonlearnables = struct;
        
        % MATLAB code that implements this invocation of the operator in
        % its graph. A string array.
        MCode = string([]);
        
        % Names of operator functions required for this node and subgraphs.
        % A string array.
        IncludedFunctionNames = string([]);
        
        % A struct of learnable parameters with their initial values
        % introduced in this node's subgraphs. (It is not known which
        % inputs to the current node are Learnable until its enclosing
        % graph is parsed.)
        SubgraphLearnables = struct;
        
        % MATLAB code for functions implementing all subgraphs below this
        % node. A string array.
        SubgraphFunctionCode = string([]);
        
        % For inspectability
        SubgraphTranslations = [];
        
        % Problems encountered during translation
        Issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
        
        % The opset version of the model that contains this node
        OpsetVersion = 1
        
        % The translation strategy class for this opset
        OpsetTranslationStrategy
        
        % The names of tensors introduced by this node. A string array.
        % Defaults to nodeProto.output. Set in translation methods if your
        % translator introduces extra tensors into the MATLAB code.
        IntroducedTensorNames = string([]);
        
        % The names of integer output tensors introduced by this node. A
        % string array.
        IntegerOutputTensorNames = string([]);
        
        % The GraphTranslation for this node's parent graph
        GraphTranslation
    end
    
    methods(Static)
        function nodeTranslation = create(nodeProto, opsetVersion, IntegerTensorNames, graphTranslation)
            % Static method to create a NodeTranslation
            nodeTranslation = nnet.internal.cnn.onnx.fcn.NodeTranslation(opsetVersion, graphTranslation);
            nodeTranslation.IntroducedTensorNames = string(nodeProto.output);       % Set default.
            nodeTranslation = translateNode(nodeTranslation, nodeProto, IntegerTensorNames);
        end
    end
    
    methods
        function this = NodeTranslation(opsetVersion, graphTranslation)
            if nargin > 0
                this.OpsetVersion = opsetVersion;
                this.GraphTranslation = graphTranslation;                
                switch this.OpsetVersion
                    case {1,2,3,4,5,6}
                        % For Opset version 6 or lower, use 
                        % Opset6TranslationStrategy
                        opSetStr = '6';
                    case num2cell(7:20)
                        % For Opset version 7 to 20, use the 
                        % corresponding Opset TranslationStrategy
                        opSetStr = num2str(this.OpsetVersion);                    
                    otherwise
                        % For higher Opset versions, use
                        % the highest fully-supported Opset version to translate. 
                        opSetStr = '20';
                end
                if graphTranslation.CoderMode
                    opSetStrategyConstructor = str2func(['nnet.internal.cnn.onnx.fcn.coder.Opset' opSetStr 'TranslationStrategy']);
                else
                    opSetStrategyConstructor = str2func(['nnet.internal.cnn.onnx.fcn.matlab.Opset' opSetStr 'TranslationStrategy']);
                end
                this.OpsetTranslationStrategy = opSetStrategyConstructor();
            end
        end
        
        function this = translateNode(this, nodeProto, IntegerTensorNames)
            % Make a NodeTranslationWarning if the node requires a function 
            % from a missing toolbox
            unlicensedIssue = warnIfOpNeedsMissingToolbox(this, nodeProto);
            % Select the translator function
            switch nodeProto.op_type
                case 'Abs'
                    translatorFcn = @translateAbs;
                case 'Acos'
                    translatorFcn = @translateAcos;
                case 'Add'
                    translatorFcn = @translateAdd;
                case 'And'
                    translatorFcn = @translateAnd;
                case 'ArgMax'
                    translatorFcn = @translateArgMax;
                case 'AveragePool'
                    translatorFcn = @translateAveragePool;
                case 'Bernoulli'
                    translatorFcn = @translateBernoulli;
                case 'BatchNormalization'
                    translatorFcn = @translateBatchNormalization;
                case 'Cast'
                    translatorFcn = @translateCast;
                case 'CategoryMapper'
                    translatorFcn = @translateCategoryMapper;
                case 'Ceil'
                    translatorFcn = @translateCeil;
                case 'Clip'
                    translatorFcn = @translateClip;
                case 'Compress'
                    translatorFcn = @translateCompress;
                case 'Concat'
                    translatorFcn = @translateConcat;
                case 'Constant'
                    translatorFcn = @translateConstant;
                case 'ConstantOfShape'
                    translatorFcn = @translateConstantOfShape;
                case 'Conv'
                    translatorFcn = @translateConv;
                case 'ConvTranspose'
                    translatorFcn = @translateConvTranspose;
                case 'Cos'
                    translatorFcn = @translateCos;
                case 'CumSum'
                    translatorFcn = @translateCumSum;
                case 'DepthToSpace'
                    translatorFcn = @translateDepthToSpace;
                case 'Div'
                    translatorFcn = @translateDiv;
                case 'Dropout'
                    translatorFcn = @translateDropout;
                case 'Elu'
                    translatorFcn = @translateElu;
                case 'Equal'
                    translatorFcn = @translateEqual;
                case 'Erf'
                    translatorFcn = @translateErf;
                case 'Exp'
                    translatorFcn = @translateExp;
                case 'Expand'
                    translatorFcn = @translateExpand;
                case 'EyeLike'
                    translatorFcn = @translateEyeLike;
                case 'Flatten'
                    translatorFcn = @translateFlatten;
                case 'Floor'
                    translatorFcn = @translateFloor;
                case 'Gather'
                    translatorFcn = @translateGather;
                case 'GatherElements'
                    translatorFcn = @translateGatherElements;
                case 'GatherND'
                    translatorFcn = @translateGatherND;
                case 'Gelu'
                    translatorFcn = @translateGelu;
                case 'Gemm'
                    translatorFcn = @translateGemm;
                case 'GlobalAveragePool'
                    translatorFcn = @translateGlobalAveragePool;
                case 'GlobalMaxPool'
                    translatorFcn = @translateGlobalMaxPool;
                case 'Greater'
                    translatorFcn = @translateGreater;
                case 'GreaterOrEqual'
                    translatorFcn = @translateGreaterOrEqual;
                case 'GridSample'
                    translatorFcn = @translateGridSample;
                case 'GRU'
                    translatorFcn = @translateGRU;
                case 'Hardmax'
                    translatorFcn = @translateHardmax;
                case 'HardSigmoid'
                    translatorFcn = @translateHardSigmoid;
                case 'HardSwish'
                    translatorFcn = @translateHardSwish;
                case 'Identity'
                    translatorFcn = @translateIdentity;
                case 'If'
                    translatorFcn = @translateIf;
                case 'ImageScaler'
                    translatorFcn = @translateImageScaler;
                case 'InstanceNormalization'
                    translatorFcn = @translateInstanceNormalization;
                case 'LayerNormalization'
                    translatorFcn = @translateLayerNormalization;
                case 'LeakyRelu'
                    translatorFcn = @translateLeakyRelu;
                case 'Less'
                    translatorFcn = @translateLess;
                case 'LessOrEqual'
                    translatorFcn = @translateLessOrEqual;
                case 'Log'
                    translatorFcn = @translateLog;
                case 'LogSoftmax'
                    translatorFcn = @translateLogSoftmax;                    
                case 'Loop'
                    translatorFcn = @translateLoop;
                case 'LRN'
                    translatorFcn = @translateLRN;
                case 'LSTM'
                    translatorFcn = @translateLSTM;
                case 'MatMul'
                    translatorFcn = @translateMatMul;
                case 'Max'
                    translatorFcn = @translateMax;
                case 'MaxPool'
                    translatorFcn = @translateMaxPool;
                case 'Min'
                    translatorFcn = @translateMin;
                case 'Mod'
                    translatorFcn = @translateMod;
                case 'Mul'
                    translatorFcn = @translateMul;
                case 'Neg'
                    translatorFcn = @translateNeg;
                case 'NonMaxSuppression'
                    translatorFcn = @translateNonMaxSuppression;
                case 'NonZero'
                    translatorFcn = @translateNonZero;
                case 'Not'
                    translatorFcn = @translateNot;
                case 'OneHot'
                    translatorFcn = @translateOneHot;
                case 'Or'
                    translatorFcn = @translateOr;
                case 'Pad'
                    translatorFcn = @translatePad;
                case 'PRelu'
                    translatorFcn = @translatePRelu;
                case 'Pow'
                    translatorFcn = @translatePow;
                case 'RandomUniformLike'
                    translatorFcn = @translateRandomUniformLike;
                case 'RandomUniform'
                    translatorFcn = @translateRandomUniform;
                case 'Range'
                    translatorFcn = @translateRange;
                case 'Reciprocal'
                    translatorFcn = @translateReciprocal;
                case 'ReduceL1'
                    translatorFcn = @translateReduceL1;
                case 'ReduceL2'
                    translatorFcn = @translateReduceL2;
                case 'ReduceLogSum'
                    translatorFcn = @translateReduceLogSum;
                case 'ReduceLogSumExp'
                    translatorFcn = @translateReduceLogSumExp;
                case 'ReduceMax'
                    translatorFcn = @translateReduceMax;
                case 'ReduceMean'
                    translatorFcn = @translateReduceMean;
                case 'ReduceMin'
                    translatorFcn = @translateReduceMin;
                case 'ReduceProd'
                    translatorFcn = @translateReduceProd;
                case 'ReduceSum'
                    translatorFcn = @translateReduceSum;
                case 'ReduceSumSquare'
                    translatorFcn = @translateReduceSumSquare;
                case 'Relu'
                    translatorFcn = @translateRelu;
                case 'Reshape'
                    translatorFcn = @translateReshape;
                case 'Resize'
                    translatorFcn = @translateResize;
                case 'RoiAlign'
                    translatorFcn = @translateRoiAlign;
                case 'Round'
                    translatorFcn = @translateRound;
                % case 'Scan'
                %     if this.OpsetVersion >= 9
                %         translatorFcn = @translateScan;
                %     else
                %         translatorFcn = @translateUnsupportedONNXLayers;
                %         this.Issues = nnet.internal.cnn.onnx.NodeTranslationError(nodeProto, ...
                %             message("nnet_cnn_onnx:onnx:UnsupportedOperator", nodeProto.op_type));
                %     end
                case 'Scatter'
                    translatorFcn = @translateScatter;
                case 'ScatterElements'
                    translatorFcn = @translateScatterElements;
                case 'ScatterND'
                    translatorFcn = @translateScatterND;
                case 'SequenceAt'
                    translatorFcn = @translateSequenceAt;
                case 'Shape'
                    translatorFcn = @translateShape;
                case 'Sigmoid'
                    translatorFcn = @translateSigmoid;
                case 'Sign'
                    translatorFcn = @translateSign;
                case 'Sin'
                    translatorFcn = @translateSin;
                case 'Size'
                    translatorFcn = @translateSize;
                case 'Slice'
                    translatorFcn = @translateSlice;
                case 'Softmax'
                    translatorFcn = @translateSoftmax;
                case 'Softplus'
                    translatorFcn = @translateSoftplus;
                case 'SpaceToDepth'
                    translatorFcn = @translateSpaceToDepth;
                case 'Split'
                    translatorFcn = @translateSplit;
                case 'SplitToSequence'
                    translatorFcn = @translateSplitToSequence;
                case 'Sqrt'
                    translatorFcn = @translateSqrt;
                case 'Squeeze'
                    translatorFcn = @translateSqueeze;
                case 'Sub'
                    translatorFcn = @translateSub;
                case 'Sum'
                    translatorFcn = @translateSum;
                case 'Tan'
                    translatorFcn = @translateTan;
                case 'Tanh'
                    translatorFcn = @translateTanh;
                case 'Tile'
                    translatorFcn = @translateTile;
                case 'TopK'
                    translatorFcn = @translateTopK;
                case 'Transpose'
                    translatorFcn = @translateTranspose;
                case 'Unsqueeze'
                    translatorFcn = @translateUnsqueeze;
                case 'Upsample'
                    translatorFcn = @translateUpsample;
                case 'Where'
                    translatorFcn = @translateWhere;
                case 'Xor'
                    translatorFcn = @translateXor;   
                otherwise
                    translatorFcn = @translateUnsupportedONNXLayers;
                    this.Issues = nnet.internal.cnn.onnx.NodeTranslationError(nodeProto, ...
                        message("nnet_cnn_onnx:onnx:UnsupportedOperator", nodeProto.op_type));
            end
            % Run the translator
            try
                this = translatorFcn(this.OpsetTranslationStrategy, this, nodeProto, IntegerTensorNames);
            catch ME
                this = translateUnsupportedONNXLayers(this.OpsetTranslationStrategy, this, nodeProto);
                if (strcmp(ME.identifier,'nnet_cnn_onnx:onnx:InitializerDataTypeUnsupported'))
                    rethrow(ME)
                end
                this.Issues = [this.Issues, nnet.internal.cnn.onnx.NodeTranslationError(nodeProto, ...
                    message("nnet_cnn_onnx:onnx:UnsupportedOperator", nodeProto.op_type))];                
            end
            this.Issues = [this.Issues unlicensedIssue];
            % If the translator failed, run the placeholder translator if
            % we haven't already
            if ~isequal(translatorFcn, @translateUnsupportedONNXLayers)
                issuesWithPlaceholdersIdx = arrayfun(@(x) isa(x, 'nnet.internal.cnn.onnx.NodeTranslationError'), this.Issues);
                issuesWithPlaceholders = this.Issues(issuesWithPlaceholdersIdx);
                if ~isempty(issuesWithPlaceholders)
                    this = translateUnsupportedONNXLayers(this.OpsetTranslationStrategy, this, nodeProto);
                end
            end
        end
        
        function issue = warnIfOpNeedsMissingToolbox(~, nodeProto)
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            switch nodeProto.op_type
                case {'DepthToSpace','Resize','SpaceToDepth', 'Upsample'}
                    % Image processing toolbox
                    if ~nnet.internal.cnn.onnx.isInstalledIPT
                        iptMsg = message('nnet_cnn_onnx:onnx:noIPTForOperator', nodeProto.op_type);
                        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(nodeProto, iptMsg);
                    end
                case {'NonMaxSuppression', 'RoiAlign'}
                    % Computer vision toolbox
                    if ~nnet.internal.cnn.onnx.isInstalledCVST
                        cvtMsg = message('nnet_cnn_onnx:onnx:noCVTForOperator', nodeProto.op_type);
                        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(nodeProto, cvtMsg);
                    end
            end
        end
    end
end
