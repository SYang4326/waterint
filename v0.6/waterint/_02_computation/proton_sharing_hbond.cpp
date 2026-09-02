#include "../chemistry.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
double mi(double d, double L, bool p) { return p ? d - std::nearbyint(d / L) * L : d; }
}

// Unique nearest-O assignment, intentionally kept beside the PES kernel: a
// hydrogen must be assigned before donor/acceptor species are selected.
extern "C" int waterint_nearest_oh_assignment(
    const double* oxygen, std::size_t no, const double* hydrogen, std::size_t nh,
    double cutoff, const double* cell, const std::uint8_t* pbc, std::size_t capacity,
    std::int64_t* counts, std::int64_t* matrix) {
    if (!oxygen || !hydrogen || !cell || !pbc || !counts || !matrix || cutoff <= 0 || capacity == 0) return 1;
    bool periodic[3]={pbc[0]!=0,pbc[1]!=0,pbc[2]!=0}; for(std::size_t o=0;o<no;++o) counts[o]=0;
    const double c2=cutoff*cutoff;
    for(std::size_t h=0;h<nh;++h){ double best=c2; std::size_t bo=no;
        for(std::size_t o=0;o<no;++o){ double r2=0; for(int q=0;q<3;++q){double x=mi(hydrogen[3*h+q]-oxygen[3*o+q],cell[q],periodic[q]);r2+=x*x;} if(r2<best){best=r2;bo=o;} }
        if(bo<no){if(static_cast<std::size_t>(counts[bo])>=capacity)return 2; matrix[bo*capacity+counts[bo]++]=static_cast<std::int64_t>(h);}
    } return 0;
}

// Accumulate unit-weight O-O pairs passing the common hydrogen-bond criterion.
// The first five histogram blocks are L1-L1 CN=0..4 (CN=0 is normally empty);
// the final block is used for unclassified/interlayer pairs.
extern "C" int waterint_proton_hbond_accumulate(
    const double* donor, std::size_t nd, const double* acceptor, std::size_t na,
    const double* hydrogens, std::size_t nh, const std::int64_t* hcounts,
    const std::int64_t* hmatrix, std::size_t hcap, const double* cell,
    const std::uint8_t* pbc, double oo_min, double oo_max, double angle_min,
    const double* delta_edges, std::size_t ndelta, const double* oo_edges,
    std::size_t noo, double* hist, std::int64_t* pair_count,
    std::int64_t* acceptor_count) {
    if (!donor || !acceptor || !hydrogens || !hcounts || !hmatrix || !cell ||
        !pbc || !delta_edges || !oo_edges || !hist || !pair_count || !acceptor_count ||
        !hcap || ndelta == 0 || noo == 0 || oo_max <= oo_min || angle_min < 0 || angle_min > 180)
        return 1;
    bool periodic[3] = {pbc[0] != 0, pbc[1] != 0, pbc[2] != 0};
    const double rad = 57.29577951308232;
    for (std::size_t a = 0; a < na; ++a) {
        std::vector<std::pair<std::size_t, std::size_t>> qualified; // donor,H
        for (std::size_t d = 0; d < nd; ++d) {
            if (hcounts[d] < 0 || static_cast<std::size_t>(hcounts[d]) > hcap) return 2;
            double ov[3]; for (int q=0;q<3;++q) ov[q]=mi(acceptor[3*a+q]-donor[3*d+q],cell[q],periodic[q]);
            double oo2=ov[0]*ov[0]+ov[1]*ov[1]+ov[2]*ov[2];
            double oo=std::sqrt(oo2); if (oo < oo_min || oo >= oo_max) continue;
            const auto* row=hmatrix+d*hcap; bool found=false; std::size_t best=0; double bestsum=1e300;
            for (std::int64_t j=0;j<hcounts[d];++j) {
                auto hi=row[j]; if (hi<0 || static_cast<std::size_t>(hi)>=nh) return 3;
                double vh[3], va[3]; for(int q=0;q<3;++q){ vh[q]=mi(hydrogens[3*hi+q]-donor[3*d+q],cell[q],periodic[q]); va[q]=mi(hydrogens[3*hi+q]-acceptor[3*a+q],cell[q],periodic[q]); }
                double dh2=vh[0]*vh[0]+vh[1]*vh[1]+vh[2]*vh[2], ha2=va[0]*va[0]+va[1]*va[1]+va[2]*va[2];
                if (!(dh2>0&&ha2>0)) continue;
                double c=(vh[0]*va[0]+vh[1]*va[1]+vh[2]*va[2])/std::sqrt(dh2*ha2); c=std::max(-1.0,std::min(1.0,c));
                if (std::acos(c)*rad < angle_min) continue;
                double sum=std::sqrt(dh2)+std::sqrt(ha2); if(sum<bestsum){bestsum=sum;best=static_cast<std::size_t>(hi);found=true;}
            }
            if(found) qualified.emplace_back(d,best);
        }
        std::size_t cn=qualified.size(); if(cn>4) cn=4; if (cn > 0) *acceptor_count += 1;
        for (auto [d,hi]: qualified) {
            double ov[3]; for(int q=0;q<3;++q)ov[q]=mi(acceptor[3*a+q]-donor[3*d+q],cell[q],periodic[q]);
            double oo=std::sqrt(ov[0]*ov[0]+ov[1]*ov[1]+ov[2]*ov[2]); double unit[3]={ov[0]/oo,ov[1]/oo,ov[2]/oo};
            double vh[3]; for(int q=0;q<3;++q)vh[q]=mi(hydrogens[3*hi+q]-donor[3*d+q],cell[q],periodic[q]);
            double dh=std::sqrt(vh[0]*vh[0]+vh[1]*vh[1]+vh[2]*vh[2]); double va[3]; for(int q=0;q<3;++q)va[q]=mi(hydrogens[3*hi+q]-acceptor[3*a+q],cell[q],periodic[q]); double ha=std::sqrt(va[0]*va[0]+va[1]*va[1]+va[2]*va[2]);
            double delta=dh-ha; auto di=std::upper_bound(delta_edges,delta_edges+ndelta+1,delta)-delta_edges-1; auto oi=std::upper_bound(oo_edges,oo_edges+noo+1,oo)-oo_edges-1;
            if(di<0 || oi<0 || static_cast<std::size_t>(di)>=ndelta || static_cast<std::size_t>(oi)>=noo) continue;
            std::size_t block=cn; hist[(block*ndelta+static_cast<std::size_t>(di))*noo+static_cast<std::size_t>(oi)] += 1.0; *pair_count += 1;
        }
    }
    return 0;
}
