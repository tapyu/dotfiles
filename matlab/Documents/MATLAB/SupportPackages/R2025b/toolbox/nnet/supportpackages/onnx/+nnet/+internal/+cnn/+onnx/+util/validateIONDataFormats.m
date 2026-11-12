function formatString = validateIONDataFormats(format, NVPName)
% 'format' specifies the format strings of multiple network inputs. If
% format is non-empty, it must be one of the following: (1) A single
% supported format string or char array, such as "SSCB" or 'SSCB'. (2) A
% string array where each element is either "" or a supported string. (3) A
% cell array where each element is either empty (of any datatype) or a
% supported char array. For example, {'SSCB', []} is supported, even though
% [] is of type double, not char.
% The output is always a string (array).

%   Copyright 2021-2022 The MathWorks, Inc.    

if isempty(format)
    formatString = string.empty;
else
    % If it's a cell array, convert empties to char type:
    if iscell(format)
        for i=1:numel(format)
            if isempty(format{i})
                format{i} = '';
            end
        end
    end
    % Check datatypes
    assert((isstring(format) || ischar(format) || iscellstr(format)) && isvector(format), message('nnet_cnn_onnx:onnx:DataFormatType', NVPName));
    % Convert string array or single char array to cell
    if isstring(format) || ischar(format)
        format = cellstr(format);
    end
    % Remove empties for content checking
    nonempties = format;
    nonempties(cellfun(@isempty, format)) = [];
    % Check that all formats are supported
    if ~isempty(nonempties)
        % Make uppercase
        fmtCell = cellstr(upper(nonempties));
        % Check chars for individual S C B T membership
        charFmt = [fmtCell{:}];
        isLabelValid = arrayfun(@(label) ismember(label, ["S", "C", "B", "T"]), charFmt);
        if any(~isLabelValid)
            error(message('nnet_cnn_onnx:onnx:DataFormatLabels', NVPName));
        end
        % Check format strings against supported ONNX list
        isSupportedFormat = cellfun(@(label) ismember(label, nnet.internal.cnn.onnx.OperatorTranslator.SupportedONNXLabels), fmtCell);
        unsupportedFormatLoc = find(~isSupportedFormat,1);
        if ~isempty(unsupportedFormatLoc)
            error(message('nnet_cnn_onnx:onnx:DataFormatStrings', NVPName, fmtCell{unsupportedFormatLoc}, ...
                iQuotifyStringArray(nnet.internal.cnn.onnx.OperatorTranslator.SupportedONNXLabels)));
        end
    end
    % Stringify
    formatString = string(format);
end
end

function str = iQuotifyStringArray(sa)
% Return a string that contains the quoted strings in the string array sa,
% surrounded by square braces.
str = "[" + join(arrayfun(@(s)'"' + s + '"', sa), ", ") + "]";
end