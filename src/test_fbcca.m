function results = test_fbcca(eeg, list_freqs, fs, num_harms, num_fbs)
% Steady-state visual evoked potentials (SSVEPs) detection using the filter
% bank canonical correlation analysis (FBCCA)-based method [1].
% 
% function results = test_fbcca(eeg, list_freqs, fs, num_harms, num_fbs)
%
% Input:
%   eeg             : Input eeg data 
%                     (# of targets, # of channels, Data length [sample])
%   list_freqs      : List for stimulus frequencies
%   fs              : Sampling frequency
%   num_harms       : # of harmonics
%   num_fbs         : # of filters in filterbank analysis
%
% Output:
%   results         : The target estimated by this method
%
% Reference:
%   [1] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao,
%       "Filter bank canonical correlation analysis for implementing a 
%        high-speed SSVEP-based brain-computer interface",
%       J. Neural Eng., vol.12, 046008, 2015.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if nargin < 3
    error('stats:test_fbcca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 5; end

fb_coefs = [1:num_fbs].^(-1.25)+0.25;

[num_targs, ~, num_smpls] = size(eeg); % 通道数,样值数,样本人数
y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms); % 获取参考信号
for targ_i = 1:1:num_targs % eeg通道数
     test_tmp = squeeze(eeg(targ_i, :, :)); % 删除长度为1的维度
     for fb_i = 1:1:num_fbs % 滤波器数
         testdata = filterbank(test_tmp, fs, fb_i); % 每次用不同的带通得到不同频段的分量
         for class_i = 1:1:num_targs % 
             refdata = squeeze(y_ref(class_i, :, :)); % 
             [~,~,r_tmp] = canoncorr(testdata', refdata'); % 计算典型相关系数
             r(fb_i, class_i) = r_tmp(1,1); % 取相关值最大的那个结果 \lambda^2
         end % class_i
     end % fb_i
     rho = fb_coefs*r; % 计算 \rho_n
    [~, tau] = max(rho);
    results(targ_i) = tau;
end % targ_i

function [ y_ref ] = cca_reference(list_freqs, fs, num_smpls, num_harms)
% Generate reference signals for the canonical correlation analysis (CCA)
% -based steady-state visual evoked potentials (SSVEPs) detection [1, 2].
%
% function [ y_ref ] = cca_reference(listFreq, fs,  nSmpls, nHarms)
% 
% Input:
%   list_freqs        : List for stimulus frequencies % 基频列表
%   fs              : Sampling frequency % 采样率
%   num_smpls          : # of samples in an epoch % 采样点数
%   num_harms          : # of harmonics % 谐波次数
%
% Output:
%   y_ref           : Generated reference signals
%                    (# of targets, 2*# of channels, Data length [sample])
%
% Reference:
%   [1] Z. Lin, C. Zhang, W. Wu, and X. Gao,
%       "Frequency Recognition Based on Canonical Correlation Analysis for 
%        SSVEP-Based BCI",
%       IEEE Trans. Biomed. Eng., 54(6), 1172-1176, 2007.
%   [2] G. Bin, X. Gao, Z. Yan, B. Hong, and S. Gao,
%       "An online multi-channel SSVEP-based brain-computer interface using
%        a canonical correlation analysis method",
%       J. Neural Eng., 6 (2009) 046002 (6pp).
%
% Masaki Nakanishi, 28-Jul-2016
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if nargin < 3 
    error('stats:cca_reference:LackOfInput',...
        'Not enough input arguments.');
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

num_freqs = length(list_freqs); % 基频数量
tidx = (1:num_smpls)/fs; % 时间序列
for freq_i = 1:1:num_freqs
    tmp = [];
    for harm_i = 1:1:num_harms % 谐波次数
        stim_freq = list_freqs(freq_i); % 本次处理的基频
        tmp = [tmp;...
            sin(2*pi*tidx*harm_i*stim_freq);...
            cos(2*pi*tidx*harm_i*stim_freq)]; % tmp [2*num_harms,num_smpls]
    end % harm_i
    y_ref(freq_i, 1:2*num_harms, 1:num_smpls) = tmp; % y_ref(基频数,2*最高次谐波阶数,采样点数)
end % freq_i