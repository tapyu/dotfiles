classdef TranslatorForUpsample < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Upsample operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator attributes
        mode
        scales
        
        % Other properties
        LayerName
        OpsetVersion
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.OpsetVersion = this.GraphProtoManager.OpsetVersion;
            
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            if this.OpsetVersion < 7
                AttributeTable = cell2table({
                    "height_scale" "FLOAT"  false   []
                    "mode"	       "STRING" true    "nearest"
                    "width_scale"  "FLOAT"  false   []
                    });
                
                [heightScale, this.mode, widthScale] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
                this.scales = [1, 1, heightScale, widthScale];
            elseif this.OpsetVersion >= 7 && this.OpsetVersion < 9
                AttributeTable = cell2table({
                    "mode"	       "STRING" true    "nearest"
                    "scales"       "FLOATS"  true   []
                    });
                
                [this.mode, this.scales] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            else
                AttributeTable = cell2table({
                    "mode"	       "STRING" true    "nearest"
                    });
                this.mode = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            end
            % Set the layer name
            this.LayerName = this.Node.name;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            [inputTensorFormats, outputTensorFormats] = propagateSISOPassthroughOp(this, direction, inputTensorFormats, outputTensorFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if ~nnet.internal.cnn.onnx.isInstalledIPT()
                issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message('nnet_cnn_onnx:onnx:noIPTForOperator', 'Upsample'));
            elseif supportedFormat(this, inputTensorFormats(1), outputTensorFormats(1))
                % 'scales' is the second input in Opset 9
                if this.OpsetVersion >= 9
                    scaleName = this.Node.input{2};
                    if isempty(initializerRawData(this.GraphProtoManager, scaleName))
                        issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node, ...
                            message("nnet_cnn_onnx:onnx:ScaleIsNotInitializer"));
                        Layer = [];
                        return;
                    else
                        this.scales = single(initializerRawData(this.GraphProtoManager, scaleName));
                    end
                end
                % Create layer
                Mode = iConvertMethodFromONNX(this, inputTensorFormats(1));
                if ~isempty(Mode)
                    if inputTensorFormats(1)=="BCSS"
                        Scales = [this.scales(end-1) this.scales(end)];
                        [Layer, issues] = constructLayer(this, 'resize2dLayer', this.LayerName, this.Node, ...
                            'Name', this.LayerName, 'Scale', Scales, 'Method', Mode, 'NearestRoundingMode', 'onnx-10','GeometricTransformMode', 'asymmetric');
                    else % "BCSSS"
                        Scales = [this.scales(end-2) this.scales(end-1) this.scales(end)];
                        [Layer, issues] = constructLayer(this, 'resize3dLayer', this.LayerName, this.Node, ...
                            'Name', this.LayerName, 'Scale', Scales, 'Method', Mode, 'NearestRoundingMode', 'onnx-10','GeometricTransformMode', 'asymmetric');
                    end
                end
            end
        end
    end
    
    methods(Access=protected)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCSS", "BCSSS"];
            if ismember(inputTensorFormat, supportedFormats) && strcmp(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end
        
        function mode = iConvertMethodFromONNX(this, inputTensorFormat)
            % Converts interpolation method string to layer type.
            mode = '';
            if strcmpi(this.mode,'linear')
                if inputTensorFormat=="BCSSS"
                    mode = 'trilinear';
                else % "BCSS
                    mode = 'bilinear';
                end
            end
        end
    end
end
