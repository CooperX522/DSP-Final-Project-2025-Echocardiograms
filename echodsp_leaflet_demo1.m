function out = echodsp_leaflet_demo1()
% Basic DSP-only pipeline for:
%   1) Preprocessing an echocardiogram video
%   2) Motion-energy based leaflet-region detection
%   3) Bright thin-structure masking for leaflets

% Usage:
%   results = echodsp_leaflet_demo();
% Edit the videoPath variable below before running.
    clc
    % USER SETTINGS
    tic
    videoPath = fullfile('GoodVideos', '0XFBC3E8226CE1D25.avi');
    %videoPath          = '\GoodVideos\0XFBC3E8226CE1D25.avi';   % <-- set this
    maxFramesToUse     = 150;                 % limit for speed
    doShowDebug        = true;                % show intermediate plots

    % Motion map parameters
    motionThrFactor    = 0.2;                 % threshold = mean + factor*std

    % Leaflet intensity / morphology parameters
    minLeafletArea     = 30;                  % minimum connected component size (pixels)

    % LOAD VIDEO
    vr = VideoReader(videoPath);
    numFrames = min(floor(vr.Duration * vr.FrameRate), maxFramesToUse);

    fprintf('Reading %d frames from %s\n', numFrames, videoPath);

    % Preallocate
    frameHeight = vr.Height;
    frameWidth  = vr.Width;
    frames      = zeros(frameHeight, frameWidth, numFrames, 'single');

    % Read and preprocess frames
    for k = 1:numFrames
        f = readFrame(vr);
        if size(f,3) == 3
            f = rgb2gray(f);
        end
        f = im2single(f);
        frames(:,:,k) = preprocessFrame(f);
    end

    
    % MOTION ENERGY MAP
    fprintf('Computing motion energy map...\n');
    motionMap = computeMotionMap(frames);

    % LEAFLET REGION BY MOTION + GEOMETRY
    fprintf('Detecting leaflet region...\n');
    T = size(frames,3);
    [leafletMotionMask, leafletBandMask] = leafletRegionFromMotion( ...
        motionMap, T, motionThrFactor);

    % LEAFLET PIXELS: BRIGHT THIN STRUCTURES IN LEAFLET REGION
    fprintf('Detecting leaflet pixels...\n');
    leafletMask = detectLeafletPixels(frames, leafletMotionMask, ...
                                       minLeafletArea);

    % WRITE LEAFLET OVERLAY VIDEO
    numFrames = size(frames,3);     % <-- define it HERE
    
    [~, baseName, ~] = fileparts(videoPath);
    videoOutPath = [baseName '_leaflet_overlay.avi'];
    vw = VideoWriter(videoOutPath, 'Motion JPEG AVI');
    vw.Quality = 100;

    open(vw);
    
    for t = 1:numFrames
        overlay = imoverlay_norm(frames(:,:,t), leafletMask(:,:,t));
        writeVideo(vw, overlay);
    end
    
    close(vw);
    fprintf('Overlay video written to %s\n', videoOutPath);


    % VISUALIZATION
    if doShowDebug
        kMid = round(numFrames/2);
        figure; 
        subplot(2,3,1);
        imshow(frames(:,:,kMid),[]);
        title(sprintf('Frame %d (preprocessed)', kMid));

        subplot(2,3,2);
        imshow(motionMap,[]);
        title('Motion energy map');

        subplot(2,3,3);
        imshow(leafletMotionMask(:,:,kMid));
        title('Leaflet motion region (mid frame)');

        subplot(2,3,4);
        imshow(leafletBandMask);
        title('Geometric leaflet band');

        subplot(2,3,5);
        overlay = imoverlay_norm(frames(:,:,kMid), leafletMask(:,:,kMid));
        imshow(overlay);
        title('Leaflet mask overlay (mid frame)');
    end

    fprintf('Done.\n');
toc
end

function fOut = preprocessFrame(fIn)
% Basic denoising + normalization, pure DSP.

    % Small Gaussian blur for speckle smoothing
    fOut = imgaussfilt(fIn, 1);

    % Normalize to zero mean, unit variance inside sector-ish region
    mu = mean(fOut(:));
    sigma = std(fOut(:)) + eps;
    fOut = (fOut - mu) / sigma;

    % Rescale to [0,1] for visualization
    fOut = mat2gray(fOut);
end



function motionMap = computeMotionMap(frames)
% Compute a global motion energy map.
% frames: H x W x T

    [H,W,T] = size(frames);
    motionEnergy = zeros(H,W,'single');

    for t = 1:(T-1)
        diffFrame = abs(frames(:,:,t+1) - frames(:,:,t)) ...
          - 0.5*abs(frames(:,:,t) - frames(:,:,max(t-1,1)));
        diffFrame(diffFrame < 0) = 0;

        motionEnergy = motionEnergy + diffFrame;
    end

    % Smooth over space
    motionMap = imgaussfilt(motionEnergy, 2);
    % Normalize to [0,1]
    motionMap = mat2gray(motionMap);
end

function [leafletMotionMask, leafletBandMask] = leafletRegionFromMotion( ...
    motionMap, T, motionThrFactor)
%   They remain in the signature to preserve call compatibility.

    % Dimensions
    [H,W] = size(motionMap);


    % HARD geometric band (logical, for indexing & masking)
    % Shifted LOWER to intersect basal LV region
    hardBand = false(H,W);
    hardBand(round(0.5*H):round(0.80*H), ...
             round(0.35*W):round(0.90*W)) = true;

    % Motion support (local evidence aggregation)
    % Neighborhood kernel (tune size if needed)
    K = ones(5,5) / 25;
    
    % Compute local support
    motionSupport = conv2(motionMap, K, 'same');
    
    % Restrict to geometric band
    motionSupport(~hardBand) = 0;
    
    % Threshold on SUPPORT, not raw motion
    vals = motionSupport(hardBand);
    m = mean(vals(:));
    s = std(vals(:)) + eps;
    
    thr = m + motionThrFactor * s;
    
    bandMotionMask2D = motionSupport > thr;
    bandMotionMask2D(~hardBand) = false;
    bandMotionMask2D = bwareaopen(bandMotionMask2D, 30);

    % Morphological cleanup
    bandMotionMask2D = bwareaopen(bandMotionMask2D, 20);
    bandMotionMask2D = imclose(bandMotionMask2D, strel('disk',2));

    % Return hard band for visualization
    leafletMotionMask = repmat(bandMotionMask2D, 1, 1, T);
    leafletBandMask = hardBand;
end



function leafletMask = detectLeafletPixels(frames, leafletMotionMask, ...
                                            minLeafletArea)
% Inside leaflet motion region, pick leaflet pixels using
% fused motion + intensity evidence (no learning).

    [H,W,T] = size(frames);
    leafletMask = false(H,W,T);

    % Weight between motion and intensity
    alpha = 0.6;   % 0.5–0.7 is a good range

    for t = 1:T
        f  = frames(:,:,t);
        lm = leafletMotionMask(:,:,t);

        if ~any(lm(:))
            continue;
        end

        % INTENSITY EVIDENCE (soft, not binary)
        intensityScore = zeros(H,W,'single');
        valsI = f(lm);
        intensityScore(lm) = mat2gray(valsI);

        % MOTION SUPPORT (local consistency)
        motionSupport = conv2(single(lm), ones(3)/9, 'same');
        motionSupport = mat2gray(motionSupport);

        % FUSED LEAFLET SCORE
        leafletScore = alpha * motionSupport + ...
                       (1 - alpha) * intensityScore;

        leafletScore(~lm) = 0;

        % Threshold fused score
        vals = leafletScore(lm);
        thr  = mean(vals(:)) + 0.3 * std(vals(:));

        bright = leafletScore > thr;

        % Cleanup + geometry
        bright = bwareaopen(bright, minLeafletArea);

        thinLeaflet = bwmorph(bright, 'skel', Inf);
        thickLeaflet = imdilate(thinLeaflet, strel('disk', 2));

        leafletMask(:,:,t) = thickLeaflet;
    end
end


function out = imoverlay_norm(grayImg, mask)
% Simple overlay: grayscale background, red mask.
% Input grayImg assumed in [0,1]; mask logical.

    grayImg = mat2gray(grayImg);
    mask = logical(mask);

    out = repmat(grayImg, [1 1 3]);
    % Red channel boosted where mask is true
    out(:,:,1) = max(out(:,:,1), 0.8 * mask);
    % Dim green/blue where mask is true
    out(:,:,2) = out(:,:,2) .* ~mask;
    out(:,:,3) = out(:,:,3) .* ~mask;
end

