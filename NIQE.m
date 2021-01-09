% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto


load modelparameters.mat
 
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

% Se pasa la direccion de la carpeta donde estan los videos 
videos_path = 'E:\ALEJANDRO LEDESMA\THESIS\CVD-2014\cvd 2014\CVD2014 I\Test1\City\';

% Direccion donde se guardan las caracteristicas extraidas
Feature_path= 'D:\JAVERIANA\THESIS\RESULTS\NIQE\Konvid\Features\';

% Direccion donde se guardan los puntajes extraidos 
Scorepath= 'D:\JAVERIANA\THESIS\RESULTS\NIQE\REPETICION\CVD2014\Scores\';

list=dir(videos_path)
for i=11:11
    %length(list)
tic
i
list(i).name

% Se carga la direccion del video a evaluar 
video = strcat(videos_path,list(i).name);    
obj = VideoReader(video);
vid = read(obj);
frames = obj.NumberOfFrames;
ST='.bmp';

% Se evalua cada frame independientemente convirtiendolos en formatos bmp 
 for x = 1:frames
      Sx=num2str(x);
      Strc=strcat(Sx,ST);
      im=vid(:,:,:,x)
featnum      = 18;
if(size(im,3)==3)
im               = rgb2gray(im);
end
im               = double(im);                
[row col]        = size(im);
block_rownum     = floor(row/blocksizerow);
block_colnum     = floor(col/blocksizecol);

im               = im(1:block_rownum*blocksizerow,1:block_colnum*blocksizecol);              
[row col]        = size(im);
block_rownum     = floor(row/blocksizerow);
block_colnum     = floor(col/blocksizecol);
im               = im(1:block_rownum*blocksizerow, ...
                   1:block_colnum*blocksizecol);               
window           = fspecial('gaussian',7,7/6);
window           = window/sum(sum(window));
scalenum         = 2;
warning('off')

feat             = [];


for itr_scale = 1:scalenum

    
mu                       = imfilter(im,window,'replicate');
mu_sq                    = mu.*mu;
sigma                    = sqrt(abs(imfilter(im.*im,window,'replicate') - mu_sq));
structdis                = (im-mu)./(sigma+1);
              
               
               
feat_scale               = blkproc(structdis,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                           [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                           @computefeature);
feat_scale               = reshape(feat_scale,[featnum ....
                           size(feat_scale,1)*size(feat_scale,2)/featnum]);
feat_scale               = feat_scale';


if(itr_scale == 1)
sharpness                = blkproc(sigma,[blocksizerow blocksizecol], ...
                           [blockrowoverlap blockcoloverlap],@computemean);
sharpness                = sharpness(:);
end


feat                     = [feat feat_scale];

im =imresize(im,0.5);

end


% Fit a MVG model to distorted patch features
distparam        = feat;
mu_distparam     = nanmean(distparam);
cov_distparam    = nancov(distparam);

% Compute quality
invcov_param     = pinv((cov_prisparam+cov_distparam)/2);
quality = sqrt((mu_prisparam-mu_distparam)* ...
    invcov_param*(mu_prisparam-mu_distparam)');
qualityScore(x,1)=quality;
Matriz_de_caracteristicas(x,:)= mu_distparam;
video_procesado{i}=list(i).name;

 end
 %score=mean(qualityScore)
 
% Se guarda la informacion extraida en archivos Mat 
 Fvideo=strcat(Feature_path,list(i).name);
 Fvideo2=strcat(Fvideo,'.mat');
 Svideo=strcat(Scorepath,list(i).name);
 Svideo2=strcat(Svideo,'.mat');
 save('videos_procesado.mat','video_procesado','-v7.3');  
 save (Svideo2,'score','-v7.3');
end

