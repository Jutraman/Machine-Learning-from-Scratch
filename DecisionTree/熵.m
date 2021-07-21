www.pudn.com > DecisionTreeID3.zip > DecisionTreeID3.m, change:2014-07-06,size:6551b





% %%%%%% ������ID3�㷨%%%%%%%%%%
clear
optdata=input('ѡ��ʵ�����ݣ�1=breastcancer.mat��2=bean.mat\n')
switch optdata
    case 1
        load('breastcancer.mat','A','T');%AΪѵ�����ݣ�TΪ�������ݣ�A,B�ĵ�һ��Ϊ���id�����һ���Ǿ������ԣ��м�����������
    case 2
        load('bean.mat','A','T');%AΪѵ�����ݣ�TΪ��������
%     case 3
%         load('play.mat','A');%AΪѵ�����ݣ�TΪ��������
end

%A1=A;A=T;T=A1;

% option=input('��Ϣ�������Ϣ�����ʣ�(1=ID3,2=C4.5)\n')

[m,n0]=size(A);
 ZM=[1:m];%��¼����
 ZMC=m; %��¼ÿ��ָ��
 ZMN=m; %��¼ÿ���������
 Counts=0; %��¼��������
 Ff{1}=[];
 F{1}=[1:n0-2];%��¼��������
 n=n0-2;

while m~=0 && n>=1
     p=length(ZMN);
%%%%%���������ض�ÿ�����������Ϣ����%%%%
%%%%%��¼���롢��������������%%%%%%%%
    CNum=[];NClass=[];CClass=[];Gain=[];ZM0=[];ZMN0=[];ZMC0=[];
    for j=1:n
        ZMa=[];ZMNa=[]; ZMCa=[];SetIna=[];Gainj=[];nclass=[];
        nh=0;
        for h=1:p
            ZMh=ZM(ZMC(h)-ZMN(h)+1:ZMC(h));%ȡ����h��������

            %%%%%%����Xh��ϵͳ��%%%%%%
            ADegh=unique(A(ZMh,n0));
            ph=length(ADegh);
            tagh=zeros(1,ZMN(h));
            ZMNh=[];
            for k=1:ph
                nn=0;
                for i=1:ZMN(h)
                    if A(ZMh(i),n0)==ADegh(k) &&  tagh(i)==0
                        nn=nn+1;
                        tagh(i)=1;
                    end
                end
                ZMNh(k)=nn;
            end
            Etrh=ShEtropy(ZMNh);

            %%%%%%%��������ص���%%%%%%%%
            ADegj=unique(A(ZMh,F{h}(j)+1));%����ȼ�
            NDegj=length(ADegj);%�ȼ�����
            nclass=[nclass,NDegj];
            tagj=zeros(1,ZMN(h));

            SetInj=[];ZMj=[]; ZMNjk=[];Etrhk=[];

            for k=1:NDegj
                %�����û�����¼ÿ����������
                ZMjk=[];
                nk=0;
                for u=1:ZMN(h)
                    if A(ZMh(u),F{h}(j)+1)==ADegj(k) && tagj(u)==0
                        nk=nk+1;
                        ZMjk(nk)=ZMh(u);
                        tagj(u)=1;
                    end
                end
                ZMj=[ZMj,ZMjk];
                ZMNjk=[ZMNjk,nk];

                %ͳ��ÿ������ھ�������g�µĻ���
                DDegjk=unique(A(ZMjk,n0));
                pk=length(DDegjk);
                tagk=zeros(1,nk);
                ZMNd=[];
                for hk=1:pk
                    nh=0;
                    for u=1:nk
                        if  A(ZMjk(u),n0)==DDegjk(hk) && tagk(u)==0
                            nh=nh+1;
                            tagk(u)=1;
                        end
                    end
                    ZMNd(hk)=nh;
                end

                %������������Ϣ�أ����жϾ������ص�ֵ�Ƿ���ͬ
                Etrhk(k)=ShEtropy(ZMNd);
                if  Etrhk(k)==0
%         if isequal(A(ZMjk,n),A(ZMjk(1),n)*ones(nk,1))
                    SetInj(k)=1;
                else
                    SetInj(k)=0;
                end
            end
            SetIna=[SetIna,SetInj];
            Etrhj=(ZMNjk/ZMN(h))*Etrhk';%����ÿ�����������������·�����
            Gainjh=Etrh-Etrhj;
            ZMa=[ZMa,ZMj];
            ZMNa=[ZMNa,ZMNjk];
            Gainj=[Gainj,Gainjh];
        end
        Gain(:,j)=Gainj';
        SetIn{j}=SetIna;
        ZM0{j}=ZMa;
        ZMN0{j}=ZMNa;
        NClass{j}=nclass;
        CClass{j}=cumsum(nclass);
        ZMC0{j}=cumsum(ZMNa);
    end

    %%%%�������ֺ�����롢����¼���ָ���¼��%%%%
    ZMw=[];ZMNw=[];ZMCw=[];SetInw=[];fj=[];Ffw=[];Fw=[];

    for h=1:p
        [gamax,mj]=max(Gain(h,:));
        ZMhm=ZMC(h)-ZMN(h)+1:ZMC(h);
        ZMw=[ZMw,ZM0{mj}(ZMhm)];%�û�������
        ZMNhm=CClass{mj}(h)-NClass{mj}(h)+1:CClass{mj}(h);
        ZMNw=[ZMNw,ZMN0{mj}(ZMNhm)];%��������
        ZMCw=[ZMCw,ZMC0{mj}(ZMNhm)];%�ָ��
        SetInw=[SetInw,SetIn{mj}(ZMNhm)];%��¼�Ƿ�ΪҶ��
        fj=[fj,mj*ones(1,NClass{mj}(h))];
        Ffh=[Ff{h} F{h}(mj)];
        F{h}(mj)=[];%���������ص���δ��������ɾȥ
        Ffw=[Ffw;ones(NClass{mj}(h),1)*Ffh];%��������
        Fw=[Fw;ones(NClass{mj}(h),1)*F{h}];%δ������
    end

    %%%%%������������%%%%%%%
    FfNum=length(Ffh);%ǰ���ظ���
    nj=length(fj);%�ܷ������

    Del=[];Deln=[];nw=0;ZMNwc=[];Ffwc=[];Fwc=[];

    for i=1:nj
        SI=SetInw(i);
        if  SI==1
            Counts=Counts+1;
            Si=ZMw(ZMCw(i));
            for j=1:FfNum
               InfSent{Counts,j}=[Ffw(i,j),A(Si,Ffw(i,j)+1)];%����ǰ��
            end
            InfSent{Counts,FfNum+1}=A(Si,n0);%������
            ISFfNum(Counts,1)=FfNum;%����ǰ�����ظ���
            Del=[Del,ZMCw(i)-ZMNw(i)+1:ZMCw(i)]; %�������ж������룬�����´η���ǰɾ��
            Deln=[Deln,nj];
        else
            nw=nw+1;
            Ffwc{nw}=Ffw(i,:);
            Fwc{nw}=Fw(i,:);
            ZMNwc(nw)=ZMNw(i);
        end
    end

    %��ʼ��

    ZMw(Del)=[];
    m=length(ZMw');
    n=n-1;
    Ff=Ffwc;
    F=Fwc;
    ZM=ZMw;
    ZMN=ZMNwc;
    ZMC=cumsum(ZMN);
end


%���Լ�testset
[m1,n1]=size(T);
SN=length(ISFfNum); %��������
Dis=[];
dt=0;td=0;wd=0;
for i=1:m1
    non=0;
    for j=1:SN
        ss=0;
        for k=1:ISFfNum(j)
            %��֤����ǰ��
            if T(i,InfSent{j,k}(1)+1)==InfSent{j,k}(2)
                ss=ss+1;
            end
        end
        if ss==ISFfNum(j)
            dt=dt+1; %ʶ���
            ED(dt,1)=i;
            ED(dt,2)=InfSent{j,ISFfNum(j)+1};
            if  ED(dt,2)==T(i,n1)
                td=td+1; %ʶ����ȷ��
                ETD(td,:)=ED(dt,:);
            else
                wd=wd+1;%ʶ������
                EWD(wd,:)=ED(dt,:);
            end
            break;
        else
            non=non+1;
        end
    end
    if non==SN   %δʶ��
        Dis=[Dis,i];
    end
end

% DtPercent=dt/m1 %ʶ��ٷֱ�
DtTrPercent=td/m1 %��ʶ��
% DtWrPercent=wd/dt  %����ʶ��
% DisPercent=(m1-dt)/m1 %δʶ��ٷֱ�

 %InfSent��ȡ�Ĺ���������[2,3] [4 2] 1�����ʾ����a2Ϊ3������a4Ϊ2ʱ����������gΪ1


 %��Ϣ�ؼ���
function etr=ShEtropy(P)

P=P/sum(P);
m=length(P);
for i=1:m
    if P(i)==0
        ETR(i)=0;
    else
        ETR(i)=-P(i)*log2(P(i));
    end
end
etr=sum(ETR);






















 
