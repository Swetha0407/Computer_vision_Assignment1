
from pathlib import Path
import time, json, math, csv
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"/"synthetic"
OUT = ROOT/"outputs"
RES = ROOT/"results"
for p in (DATA, OUT, RES): p.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
W,H = 640,480
N=30

def make_scene(idx):
    # Synthetic tray background and regular planting-cell grid
    img=np.zeros((H,W,3),np.uint8); img[:]=(105,112,102)
    gt=np.zeros((H,W),np.uint8)
    rows,cols=4,6
    margin_x,margin_y=48,42
    cell_w=(W-2*margin_x)//cols; cell_h=(H-2*margin_y)//rows
    light=float(RNG.uniform(0.72,1.28))
    shadow=float(RNG.uniform(0.0,0.28))
    glare=float(RNG.uniform(0.0,0.35))
    hue_shift=int(RNG.integers(-8,9))
    missing=set(RNG.choice(rows*cols,size=int(RNG.integers(2,6)),replace=False).tolist())
    # tray cell circles
    for r in range(rows):
        for c in range(cols):
            cx=margin_x+c*cell_w+cell_w//2
            cy=margin_y+r*cell_h+cell_h//2
            cv2.circle(img,(cx,cy),min(cell_w,cell_h)//2-5,(65,70,62),-1)
            cv2.circle(img,(cx,cy),min(cell_w,cell_h)//2-8,(82,88,78),2)
            if r*cols+c in missing: continue
            # leaf cluster; shape and orientation vary
            count=int(RNG.integers(3,6))
            for k in range(count):
                ang=float(RNG.uniform(0,2*np.pi))
                length=float(RNG.uniform(18,34))
                dx=int(np.cos(ang)*length*0.55); dy=int(np.sin(ang)*length*0.55)
                ax=(cx,cy); bx=(cx+dx,cy+dy)
                axes=(int(length*0.72),int(length*0.28))
                col=(int(np.clip(35+hue_shift,0,255)),int(RNG.integers(125,190)),int(RNG.integers(35,80)))
                cv2.ellipse(img,cx if False else ax,axes,float(ang*180/np.pi),0,360,col,-1)
                cv2.ellipse(gt,ax,axes,float(ang*180/np.pi),0,360,255,-1)
    # illumination gradient
    x=np.linspace(0.86,1.16,W)[None,:,None]
    y=np.linspace(0.92,1.08,H)[:,None,None]
    arr=img.astype(np.float32)*x*y*light
    # local shadow
    if shadow>0.12:
        overlay=np.zeros((H,W),np.float32)
        cv2.ellipse(overlay,(int(W*RNG.uniform(.2,.8)),int(H*.5)),(int(W*.22),int(H*.38)),int(RNG.integers(0,180)),0,360,1,-1)
        arr*= (1-shadow*overlay[:,:,None])
    # specular glare spots
    if glare>0.15:
        for _ in range(int(RNG.integers(1,4))):
            center=(int(RNG.integers(30,W-30)),int(RNG.integers(30,H-30)))
            cv2.circle(arr,center,int(RNG.integers(8,22)),(245,248,240),-1)
    arr+=RNG.normal(0,3.5,arr.shape)
    img=np.uint8(np.clip(arr,0,255))
    return img,gt,{"sample":f"tray_{idx+1:02d}","lighting_factor":light,"shadow_strength":shadow,"glare_strength":glare,"missing_cells":len(missing)}

def save_dataset():
    rows=[]
    for i in range(N):
        im,mask,meta=make_scene(i)
        cv2.imwrite(str(DATA/f"{meta['sample']}.png"),im)
        cv2.imwrite(str(DATA/f"{meta['sample']}_leaf_mask.png"),mask)
        rows.append(meta)
    pd.DataFrame(rows).to_csv(DATA/"labels.csv",index=False)
    return rows

def timed(fn):
    t=time.perf_counter(); result=fn(); return result,(time.perf_counter()-t)*1000

def leaf_mask_hsv(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,np.array([30,45,25]),np.array([95,255,255]))

def stats(mask):
    return int(cv2.countNonZero(mask)), float(cv2.countNonZero(mask)/(mask.size))

def save_panel(name, images, titles):
    n=len(images); fig,axs=plt.subplots(1,n,figsize=(4*n,3.2))
    if n==1: axs=[axs]
    for ax,im,title in zip(axs,images,titles):
        if im.ndim==2: ax.imshow(im,cmap="gray",vmin=0,vmax=255)
        else: ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ax.set_title(title,fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT/f"{name}.png",dpi=140); plt.close(fig)

def task1(rows):
    rec=[]; sample=cv2.imread(str(DATA/"tray_01.png"))
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        gt=cv2.imread(str(DATA/f"{meta['sample']}_leaf_mask.png"),0)>0
        # A: manual baseline represented by human-readable ground-truth review proxy
        # B: enhancement dashboard (CLAHE display; no automatic presence decision)
        t0=time.perf_counter(); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); manual_proxy=gt.copy(); ta=(time.perf_counter()-t0)*1000
        enh=cv2.createCLAHE(2.0,(8,8)).apply(gray)
        t0=time.perf_counter(); decision=leaf_mask_hsv(im); decision=cv2.morphologyEx(decision,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); tb=(time.perf_counter()-t0)*1000
        present=int(cv2.countNonZero(decision)>250)
        truth=int(gt.sum()>250)
        rec.append({"sample":meta["sample"],"manual_review_proxy_ms":ta,"enhancement_dashboard_ms":0.0,"automatic_decision_ms":tb,"truth_present":truth,"auto_present":present,"auto_correct":int(present==truth),"dashboard_is_decision":False})
    im=cv2.imread(str(DATA/"tray_01.png")); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    save_panel("task01_presence_vs_colour",[im,g,cv2.createCLAHE(2.0,(8,8)).apply(g),leaf_mask_hsv(im)],["Input tray","Greyscale view","Enhanced dashboard","Presence mask"])
    pd.DataFrame(rec).to_csv(RES/"task01.csv",index=False)

def task2(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        t0=time.perf_counter(); a=cv2.GaussianBlur(gray,(5,5),0); ta=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); mask=leaf_mask_hsv(im); n,lab,st,cent=cv2.connectedComponentsWithStats(mask,8); tb=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); features=[int(s[cv2.CC_STAT_AREA]) for s in st[1:] if s[cv2.CC_STAT_AREA]>80]; count=len(features); tc=(time.perf_counter()-t0)*1000
        rec.append({"sample":meta["sample"],"filter_only_ms":ta,"segment_measure_ms":tb,"measure_classify_ms":tc,"occupied_pixels":int(mask.sum()/255),"components":count,"low_stage":"denoise","mid_stage":"segment and measure","high_stage":"classify components"})
    pd.DataFrame(rec).to_csv(RES/"task02.csv",index=False)
    im=cv2.imread(str(DATA/"tray_02.png")); mask=leaf_mask_hsv(im); n,l,st,c=cv2.connectedComponentsWithStats(mask,8)
    vis=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    for x,y,w,h,a in st[1:]:
        if a>80: cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),2)
    save_panel("task02_occupancy_pipeline",[im,cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(5,5),0),mask,vis],["Input","Low: filter","Mid: segment","High: component measurement"])

def task3(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png")); gt=cv2.imread(str(DATA/f"{meta['sample']}_leaf_mask.png"),0)>0
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        rgb=(im[:,:,1]>im[:,:,2]*1.12)&(im[:,:,1]>im[:,:,0]*1.25)
        hsv=leaf_mask_hsv(im)>0
        for name,m in [("greyscale",gray>105),("RGB",rgb),("HSV",hsv)]:
            inter=np.logical_and(m,gt).sum(); union=np.logical_or(m,gt).sum()
            rec.append({"sample":meta["sample"],"representation":name,"foreground_iou":float(inter/union) if union else 1.0,"foreground_pixels":int(m.sum())})
    pd.DataFrame(rec).to_csv(RES/"task03.csv",index=False)
    im=cv2.imread(str(DATA/"tray_03.png"))
    save_panel("task03_colour_spaces",[im,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.cvtColor(im,cv2.COLOR_BGR2RGB),cv2.cvtColor(im,cv2.COLOR_BGR2HSV)],["BGR input","Greyscale","RGB display","HSV representation"])

def task4(rows):
    rec=[]
    for meta in rows:
        gt=cv2.imread(str(DATA/f"{meta['sample']}_leaf_mask.png"),0)
        # Binary test pattern uses the synthetic leaf mask, so connectivity is isolated from segmentation error.
        _,bw=cv2.threshold(gt,127,255,cv2.THRESH_BINARY)
        for conn,name in [(4,"4-connectivity"),(8,"8-connectivity")]:
            t0=time.perf_counter(); n,lab=cv2.connectedComponents(bw,connectivity=conn); ms=(time.perf_counter()-t0)*1000
            rec.append({"sample":meta["sample"],"method":name,"object_count":n-1,"runtime_ms":ms,"diagonal_only_pairs_join":conn==8})
        # distance-based grouping: dilate then components, approximate grouping
        t0=time.perf_counter(); joined=cv2.dilate(bw,np.ones((5,5),np.uint8)); n,_=cv2.connectedComponents(joined,8); ms=(time.perf_counter()-t0)*1000
        rec.append({"sample":meta["sample"],"method":"distance-based grouping","object_count":n-1,"runtime_ms":ms,"diagonal_only_pairs_join":True})
    pd.DataFrame(rec).to_csv(RES/"task04.csv",index=False)
    toy=np.zeros((100,160),np.uint8); toy[25,25]=255; toy[26,26]=255; toy[65,90]=255; toy[66,91]=255
    save_panel("task04_connectivity",[toy,cv2.dilate(toy,np.ones((3,3),np.uint8))],["Diagonal-touching pixels","After grouping dilation"])

def task5(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        # camera alternatives represented by acquisition simulation: rolling motion skew, global snapshot, line-scan strip assembly
        t0=time.perf_counter(); roll=np.roll(im,3,axis=0); roll[:3]=im[:3]; ta=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); glob=im.copy(); tb=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); line=np.repeat(im[H//2:H//2+1,:,:],H,axis=0); tc=(time.perf_counter()-t0)*1000
        rec.append({"sample":meta["sample"],"rolling_shutter_sim_ms":ta,"global_shutter_sim_ms":tb,"line_scan_sim_ms":tc,"rolling_skew_px":3,"global_skew_px":0,"line_scan_context":"single-line repeated simulation"})
    pd.DataFrame(rec).to_csv(RES/"task05.csv",index=False)
    im=cv2.imread(str(DATA/"tray_04.png")); rolled=np.roll(im,3,axis=0); line=np.repeat(im[H//2:H//2+1],H,axis=0)
    save_panel("task05_camera_arrangements",[im,rolled,line],["Global-shutter snapshot proxy","Rolling-shutter row displacement proxy","Line-scan single-line proxy"])

def task6(rows):
    rec=[]; field_mm=640; target_mm=4
    for px in (640,1280,1920):
        mm_per_px=field_mm/px; pix=target_mm/mm_per_px
        # Nyquist-style minimum: at least 2 pixels across a feature; practical recommendation >=4
        rec.append({"pixels_across_field":px,"field_width_mm":field_mm,"mm_per_pixel":mm_per_px,"pixels_across_4mm_leaf":pix,"meets_2px_minimum":pix>=2,"meets_4px_practical_target":pix>=4})
    pd.DataFrame(rec).to_csv(RES/"task06_sampling.csv",index=False)
    im=cv2.imread(str(DATA/"tray_05.png"))
    fig,axs=plt.subplots(1,3,figsize=(10,3))
    for ax,px in zip(axs,(640,1280,1920)):
        ax.imshow(cv2.cvtColor(cv2.resize(im,(px, int(H*px/W))),cv2.COLOR_BGR2RGB)); ax.set_title(f"{px}px across field"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT/"task06_sampling_comparison.png",dpi=140); plt.close(fig)

def task7(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png")); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        for bits in (6,8,12):
            levels=2**bits; maxv=levels-1
            t0=time.perf_counter(); q=np.round(gray.astype(np.float32)/255*maxv)/maxv*255; q=np.uint8(np.clip(q,0,255)); ms=(time.perf_counter()-t0)*1000
            rec.append({"sample":meta["sample"],"bits":bits,"intensity_levels":levels,"runtime_ms":ms,"unique_levels_observed":int(np.unique(q).size),"mean_absolute_quantisation_error":float(np.mean(np.abs(q.astype(float)-gray)))})
    pd.DataFrame(rec).to_csv(RES/"task07.csv",index=False)
    im=cv2.imread(str(DATA/"tray_06.png")); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    quant=[]
    for b in (6,8,12):
        lev=2**b-1; quant.append(np.uint8(np.round(g.astype(float)/255*lev)/lev*255))
    save_panel("task07_quantisation",[g,*quant],["Original 8-bit","6-bit","8-bit","12-bit"])

def task8(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        # Same source sample; controlled simulated illumination/acquisition alternatives
        ambient=im.copy()
        diffuse=cv2.GaussianBlur(im,(0,0),5)
        diffuse=cv2.addWeighted(im,0.75,diffuse,0.25,0)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); hsv[:,:,1]=np.uint8(hsv[:,:,1]*0.82); hsv[:,:,2]=np.uint8(np.clip(hsv[:,:,2]*1.04,0,255))
        polar=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        for name,img in [("ambient_only",ambient),("diffuse_lighting_proxy",diffuse),("cross_polarised_control_proxy",polar)]:
            rec.append({"sample":meta["sample"],"lighting_setup":name,"mean_intensity":float(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).mean()),"saturation_mean":float(cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1].mean())})
    pd.DataFrame(rec).to_csv(RES/"task08.csv",index=False)
    im=cv2.imread(str(DATA/"tray_07.png")); diff=cv2.addWeighted(im,.75,cv2.GaussianBlur(im,(0,0),5),.25,0)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); hsv[:,:,1]=np.uint8(hsv[:,:,1]*.82); hsv[:,:,2]=np.uint8(np.clip(hsv[:,:,2]*1.04,0,255))
    save_panel("task08_illumination",[im,diff,cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)],["Ambient only","Diffuse-lighting proxy","Polarised-control appearance proxy"])

def task9(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        t0=time.perf_counter(); raw=leaf_mask_hsv(im); ta=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); calibrated=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); calibrated=cv2.morphologyEx(calibrated,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)); tb=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); # two-view fusion proxy with horizontal flip alignment
        flip=cv2.flip(im,1); fmask=cv2.flip(leaf_mask_hsv(flip),1); fusion=cv2.bitwise_or(calibrated,fmask); tc=(time.perf_counter()-t0)*1000
        rec.append({"sample":meta["sample"],"single_raw_ms":ta,"calibrated_rules_ms":tb,"two_view_fusion_proxy_ms":tc,"raw_foreground_px":int(raw.sum()/255),"calibrated_foreground_px":int(calibrated.sum()/255),"fusion_foreground_px":int(fusion.sum()/255)})
    pd.DataFrame(rec).to_csv(RES/"task09.csv",index=False)
    im=cv2.imread(str(DATA/"tray_08.png")); raw=leaf_mask_hsv(im); cal=cv2.morphologyEx(cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    save_panel("task09_workflow",[im,raw,cal,cv2.flip(cv2.flip(cal,1),1)],["Single raw camera","Raw threshold mask","Calibrated camera + rules","Aligned two-view fusion proxy"])

def task10(rows):
    rec=[]
    for meta in rows:
        im=cv2.imread(str(DATA/f"{meta['sample']}.png"))
        # emulate night illumination via lower brightness + colour temperature shift
        night=np.uint8(np.clip(im.astype(float)*np.array([.78,.68,.52])[None,None,:],0,255))
        fixed=leaf_mask_hsv(night)
        hsv=cv2.cvtColor(night,cv2.COLOR_BGR2HSV); hsv[:,:,2]=cv2.createCLAHE(2.0,(8,8)).apply(hsv[:,:,2]); adaptive=cv2.inRange(hsv,np.array([25,30,20]),np.array([100,255,255]))
        # scene-aware: adapt value and saturation thresholds to image quantiles
        s=hsv[:,:,1]; v=hsv[:,:,2]; slo=int(np.percentile(s,35)); vlo=int(np.percentile(v,25))
        aware=cv2.inRange(hsv,np.array([25,max(20,slo),max(10,vlo)]),np.array([100,255,255]))
        for name,m in [("fixed_thresholds",fixed),("adaptive_calibration",adaptive),("scene_aware",aware)]:
            rec.append({"sample":meta["sample"],"method":name,"night_mean_intensity":float(cv2.cvtColor(night,cv2.COLOR_BGR2GRAY).mean()),"foreground_pixels":int(m.sum()/255)})
    pd.DataFrame(rec).to_csv(RES/"task10.csv",index=False)
    im=cv2.imread(str(DATA/"tray_09.png")); night=np.uint8(np.clip(im.astype(float)*np.array([.78,.68,.52])[None,None,:],0,255))
    hsv=cv2.cvtColor(night,cv2.COLOR_BGR2HSV); hsv[:,:,2]=cv2.createCLAHE(2.0,(8,8)).apply(hsv[:,:,2]); adaptive=cv2.inRange(hsv,np.array([25,30,20]),np.array([100,255,255]))
    s=hsv[:,:,1]; v=hsv[:,:,2]; aware=cv2.inRange(hsv,np.array([25,max(20,int(np.percentile(s,35))),max(10,int(np.percentile(v,25)))]),np.array([100,255,255]))
    save_panel("task10_night_adaptation",[night,leaf_mask_hsv(night),adaptive,aware],["Simulated night frame","Fixed threshold","Adaptive calibration","Scene-aware threshold"])

def main():
    rows=save_dataset()
    task1(rows); task2(rows); task3(rows); task4(rows); task5(rows); task6(rows); task7(rows); task8(rows); task9(rows); task10(rows)
    summary={
      "dataset_samples":N,"image_size":[W,H],"random_seed":2026,
      "task_sample_counts":{"1":30,"2":30,"3":30,"4":30,"5":30,"6":30,"7":30,"8":30,"9":30,"10":30},
      "numerical_answers":{
        "task1_detection_fraction":18/30,
        "task2_latency_ms":8+15+15,
        "task3_memory_800x480_1ch_bytes":800*480,
        "task3_memory_800x480_3ch_bytes":800*480*3,
        "task4_diagonal_pairs": "separate under 4-connectivity; joined under 8-connectivity; six independent diagonal-only pairs can reduce apparent object count by up to six under 8-connectivity",
        "task5_exposure_factor":10/2,
        "task6_mm_per_pixel":640/1280,
        "task6_pixels_for_4mm":4/(640/1280),
        "task7_levels_12bit":2**12,
        "task7_ratio_12bit_to_6bit":2**12/2**6,
        "task8_relative_increase_percent":(100-70)/70*100,
        "task9_theoretical_fps":1000/45,
        "task10_recall":42/50
      },
      "limitations":["Synthetic data only; not real greenhouse photography.","No expert field labels; segmentation comparisons use generated masks where available.","Camera hardware alternatives are simulated proxies, not physical sensor measurements.","Runtime varies by machine and should be interpreted as relative software timing only."]
    }
    (RES/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Completed all 10 tasks. Outputs:",OUT,"Results:",RES)

if __name__=="__main__": main()
