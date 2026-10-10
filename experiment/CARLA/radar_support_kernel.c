#include <math.h>
#include <stdint.h>
#include <stddef.h>

static int plane_code(const double *cloud,const double *point,const double *matrix,
                      const int64_t *indices,int64_t begin,int64_t end) {
    double a=0,b=0,c=0,d=0,e=0,f=(double)(end-begin),rx=0,ry=0,rz=0;
    for(int64_t j=begin;j<end;++j) {
        int64_t k=indices[j];double x=cloud[3*k]-point[0],y=cloud[3*k+1]-point[1],z=cloud[3*k+2];
        a+=x*x;b+=x*y;c+=x;d+=y*y;e+=y;rx+=x*z;ry+=y*z;rz+=z;
    }
    double q=(a+d+f)/3.,aa=a-q,dd=d-q,ff=f-q;
    double p=sqrt((aa*aa+dd*dd+ff*ff+2*(b*b+c*c+e*e))/6.);
    if(p<1e-14)return 0;
    double determinant=aa*(dd*ff-e*e)-b*(b*ff-c*e)+c*(b*e-c*dd);
    double r=determinant/(2*p*p*p);
    r=fmax(-1.,fmin(1.,r));double phi=acos(r)/3.;
    double largest=q+2*p*cos(phi),smallest=q+2*p*cos(phi+2.0943951023931954923);
    if(smallest<=1e-12 || largest<=0)return 0;
    double condition=sqrt(largest/smallest);
    if(fabs(condition-100.)<1e-6)return 0;
    if(condition>100.)return 3;
    if(a<=1e-12)return 0;
    double l00=sqrt(a),l10=b/l00,l20=c/l00;
    double s11=d-l10*l10;if(s11<=1e-12)return 0;
    double l11=sqrt(s11),l21=(e-l20*l10)/l11;
    double s22=f-l20*l20-l21*l21;if(s22<=1e-12)return 0;
    double l22=sqrt(s22),y0=rx/l00,y1=(ry-l10*y0)/l11,y2=(rz-l20*y0-l21*y1)/l22;
    double fit2=y2/l22,fit1=(y1-l21*fit2)/l11,fit0=(y0-l10*fit1-l20*fit2)/l00;
    double square=0,max_residual=0;
    for(int64_t j=begin;j<end;++j) {
        int64_t k=indices[j];double residual=(cloud[3*k]-point[0])*fit0+(cloud[3*k+1]-point[1])*fit1+fit2-cloud[3*k+2];
        square+=residual*residual;max_residual=fmax(max_residual,fabs(residual));
    }
    double slope=sqrt(fit0*fit0+fit1*fit1),rmse=sqrt(square/f);
    double origin=fit2+fit0*(matrix[3]-point[0])+fit1*(matrix[7]-point[1]);
    double height=matrix[11]-origin,above=fabs(point[2]-fit2);
    if(fabs(slope-.25)<1e-7 || fabs(rmse-.015)<1e-7 || fabs(max_residual-.04)<1e-7
       || fabs(height-.5)<1e-7 || fabs(height-2.5)<1e-7 || fabs(above-.02)<1e-7)return 0;
    if(slope>.25 || rmse>.015 || max_residual>.04)return 3;
    if(height<.5 || height>2.5)return 4;
    if(above>.02)return 5;
    return 6;
}

int screen_support(const double *cloud, const double *points, const double *matrix, const int64_t *indices,
                   const int64_t *offsets, size_t count, size_t cloud_count, int32_t *codes) {
    for (size_t i=0; i<count; ++i) {
        int64_t begin=offsets[i], end=offsets[i+1];
        if (begin<0 || end<begin) return 1;
        codes[i]=0;
        if (end-begin<8) { codes[i]=1; continue; }
        double minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY;
        double minz=INFINITY,maxz=-INFINITY;
        int64_t low=-1,high=-1;
        for (int64_t j=begin;j<end;++j) {
            int64_t k=indices[j];
            if (k<0 || (size_t)k>=cloud_count) return 2;
            double x=cloud[3*k]-points[3*i],y=cloud[3*k+1]-points[3*i+1],z=cloud[3*k+2];
            if (x<minx) minx=x;
            if (x>maxx) maxx=x;
            if (y<miny) miny=y;
            if (y>maxy) maxy=y;
            if (z<minz) { minz=z;low=k; }
            if (z>maxz) { maxz=z;high=k; }
        }
        if (minx>=-.25 || miny>=-.25 || maxx<=.25 || maxy<=.25) {codes[i]=2;continue;}
        if (low<0 || high<0) continue;
        double dx=cloud[3*high]-cloud[3*low],dy=cloud[3*high+1]-cloud[3*low+1];
        double margin=(maxz-minz)-(.25*sqrt(dx*dx+dy*dy)+.08+1e-8);
        if (margin>1e-10) codes[i]=3;
        else codes[i]=plane_code(cloud,points+3*i,matrix,indices,begin,end);
    }
    return 0;
}
