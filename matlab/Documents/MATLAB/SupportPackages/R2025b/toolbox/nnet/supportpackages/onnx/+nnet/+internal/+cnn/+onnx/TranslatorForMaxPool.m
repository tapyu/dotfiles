classdef TranslatorForMaxPool < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX MaxPool operators into MATLAB layers
    
    % Copyright 2021-2022 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Node attributes
        auto_pad
        ceil_mode
        dilations
        kernel_shape
        pads
        storage_order
        strides
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.IsSeedOperator = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "auto_pad"          "STRING"    true    "NOTSET"
                "ceil_mode"         "INT"       true    0
                "dilations"         "INTS"      true    []
                "kernel_shape"      "INTS"      false   []
                "pads"              "INTS"      true    []
                "storage_order"     "INT"       true    []
                "strides"           "INTS"      true    []
                });
            % Parse the attributes
            [this.auto_pad, this.ceil_mode, this.dilations, this.kernel_shape, this.pads, this.storage_order, this.strides] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
            % Set the layer name
            this.LayerName = this.Node.name;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % Derive formats them from the dimensionality of kernel_shape
                kernelShape = this.kernel_shape;
                if numel(kernelShape)==1
                    inputFormats(1) = "BCT";
                    outputFormats(1) = "BCT";
                elseif numel(kernelShape)==2
                    inputFormats(1) = "BCSS";
                    outputFormats(1) = "BCSS";
                elseif numel(kernelShape)==3
                    inputFormats(1) = "BCSSS";
                    outputFormats(1) = "BCSSS";
                end
            end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if hasOneOutput(this)
                % Validate the input/output formats
                if ~supportedFormat(this, inputTensorFormats(1), outputTensorFormats(1))
                    Layer = [];
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedInputTensorFormat', inputTensorFormats, 'AveragePool'));
                    return;
                end
                
                % ceil_mode=true is not supported
                if this.ceil_mode==1
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:AttributeMustHaveValue', 'MaxPool', 'ceil_mode', '0', this.Node.name));
                    Layer = [];
                    return;
                end
                
                % dilated MaxPool is not supported
                if ~(isempty(this.dilations) || all(this.dilations == 1))
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:AttributeMustHaveValue', 'MaxPool', 'dilations', '1', this.Node.name));
                    Layer = [];
                    return;
                end
                
                % storage_order=1 is not supported (Opset >=8)
                if this.storage_order==1
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:MaxPoolStorageOrder'))];
                    Layer = [];
                    return;
                end
                
                % kernel_shape
                FilterSize = this.kernel_shape; % [h w] or [h w d]
                
                % strides
                if isempty(this.strides)
                    Strides = getDefaultStrides(this, inputTensorFormats(1));
                else
                    Strides = this.strides;
                end
                
                % Handle padding
                if ~isempty(this.pads)
                    Padding = this.pads;
                    Padding = iConvertONNXToMATLABPadding(this, Padding, inputTensorFormats(1));
                    if this.auto_pad~="NOTSET"
                        issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                            message('nnet_cnn_onnx:onnx:AutoPadAndPadDefined'))];
                    end
                else
                    switch this.auto_pad
                        case 'SAME_UPPER'
                            Padding = 'same';
                        case 'SAME_LOWER'
                            Padding = 'same';
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:AutoPadSameLower', this.LayerName))];
                        case 'VALID'
                            Padding = getDefaultPadding(this, inputTensorFormats(1));
                        case 'NOTSET'
                            % Pads is not explicitly set at this point so default is used
                            Padding = getDefaultPadding(this, inputTensorFormats(1));
                        otherwise
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:UnknownAttributeSetting', 'auto_pad'))];
                    end
                end
                
                args = {FilterSize, 'Stride', Strides, 'Padding', Padding, 'Name', this.LayerName};
                if isequal(inputTensorFormats(1), 'BCSSS')
                    [Layer, constructionIssue] = constructLayer(this, 'maxPooling3dLayer', this.LayerName, this.Node, args{:});
                elseif isequal(inputTensorFormats(1), 'BCSS')
                    [Layer, constructionIssue] = constructLayer(this, 'maxPooling2dLayer', this.LayerName, this.Node, args{:});
                else
                    [Layer, constructionIssue] = constructLayer(this, 'maxPooling1dLayer', this.LayerName, this.Node, args{:});
                end
                issues = [issues constructionIssue];
                
            end
        end
    end
    
    methods(Access=protected)
        function Padding = iConvertONNXToMATLABPadding(~, Padding, inputTensorFormat)
            if strcmp(inputTensorFormat, "BCSS")
                % ONNX:   [H_b,W_b,H_end,W_end] ==> [t l b r]
                % MATLAB: [t b l r]
                Padding = Padding([1,3,2,4]);
            elseif strcmp(inputTensorFormat,"BCSSS") % "BCSSS"
                % ONNX:   [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r k]
                % MATLAB: [t l f; b r k]
                Padding = Padding([1 2 3; 4 5 6]);
            end
        end
        
        function Padding = getDefaultPadding(~, inputTensorFormat)
            if strcmp(inputTensorFormat, "BCSS")
                Padding = [0 0 0 0]; % [t b l r]
            elseif strcmp(inputTensorFormat,"BCSSS") % "BCSSS"
                Padding = [0 0 0; 0 0 0]; % [t l f; b r k]
            else
                Padding = [0 0]; %[t l]
            end
        end
        
        function Strides = getDefaultStrides(~, inputTensorFormat)
            if strcmp(inputTensorFormat, "BCSS")
                Strides = [1 1]; % [h w]
            elseif strcmp(inputTensorFormat, "BCSSS") % "BCSSS"
                Strides = [1 1 1]; % [h w d]
            else
                Strides = 1; % [h]
            end
        end
        
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCSS", "BCT", "BCSSS"];
            if ismember(inputTensorFormat, supportedFormats) && strcmp(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end
        
        function tf = hasOneOutput(this)
            tf = numel(this.Node.output) == 1 ...
                || ~outputUsedInNetwork(this, 2);
        end
    end
end
