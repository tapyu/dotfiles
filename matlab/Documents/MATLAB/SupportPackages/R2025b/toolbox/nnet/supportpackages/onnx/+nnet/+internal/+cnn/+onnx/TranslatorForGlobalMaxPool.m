classdef TranslatorForGlobalMaxPool < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX GlobalMaxPool operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
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
            if supportedFormat(this, inputTensorFormats(1), outputTensorFormats(1))
                switch inputTensorFormats(1)
                    case {"BCSS", "TBCSS", "CSS", "SSC"}
                        [Layer, issues] = constructLayer(this, "globalMaxPooling2dLayer", this.Node.name, this.Node, 'Name', this.Node.name);
                    case {"BCSSS", "TBCSSS","CSSS", "SSSC"}
                        [Layer, issues] = constructLayer(this, "globalMaxPooling3dLayer", this.Node.name, this.Node, 'Name', this.Node.name);
                    otherwise
                        [Layer, issues] = constructLayer(this, "globalMaxPooling1dLayer", this.Node.name, this.Node, 'Name', this.Node.name);
                end
            end
        end
    end
    
    methods(Access=private)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCSS", "TBCSS", "CSS", "SSC", ...
                                "BCSSS","TBCSSS","CSSS", "SSSC", ...
                                "BCT", "TBC", "BTC"];
            if ismember(inputTensorFormat, supportedFormats) && isequal(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end
    end
end
