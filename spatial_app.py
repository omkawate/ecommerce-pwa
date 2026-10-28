"""Presentation workspace. Existing model and preprocessing are used unchanged."""
import os
from pathlib import Path
import io
import json
import hashlib
import tempfile
import threading
from datetime import datetime, timezone
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
st.set_page_config(page_title='TerraScope | Spatial Intelligence', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:radial-gradient(ellipse at top right,#183b40 0%,#09141f 55%);color:#e6f0f5}
[data-testid="stSidebar"] {background:#0c1b29;border-right:1px solid #294351}
h1,h2,h3 {letter-spacing:-.04em} h1 {font-size:3rem!important}
[data-testid="stMetric"] {background:#112634;border:1px solid #2a4855;border-radius:16px;padding:18px}
.stButton>button {border-radius:10px;border:1px solid #3f6972}
[data-testid="stImage"] {background:#071019;border:1px solid #294351;border-radius:14px;padding:10px}
.block-container {padding-top:2.5rem;padding-bottom:3rem}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def engine():
    from src.predict import SatellitePredictor
    predictor = SatellitePredictor(str(ROOT / 'config.json'))
    predictor.load_model()
    return predictor, threading.Lock()

def classify(data, suffix):
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError('Use images below 25 megapixels.')
        image.verify()
    predictor, lock = engine()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / ('input' + suffix)
        path.write_bytes(data)
        with lock:
            result = predictor.predict_single(path)
    return {'predicted_class': result['predicted_class'], 'confidence': float(result['confidence']),
            'all_probabilities': result['all_probabilities']}

st.session_state.setdefault('results', [])
with st.sidebar:
    st.title('◈ TerraScope')
    st.caption('EARTH OBSERVATION WORKSPACE')
    section = st.radio('Workspace', ['Scene analysis', 'Mission history', 'Model evidence'])
    st.divider()
    threshold = st.slider('Review threshold (%)', 0, 100, 80)
    st.caption('Flags uncertain predictions for review; does not change model outputs.')
    st.markdown('**EuroSAT · RGB imagery**  \n10 land-cover classes · 64 × 64 input')
    st.caption('Whole-image classification. No object localization or geographic coordinates are inferred.')
    if st.button('Clear session history'):
        st.session_state.results = []
        st.rerun()

st.caption('TERRASCOPE / SPATIAL INTELLIGENCE')
st.title(section)
st.markdown('Explore satellite imagery. Understand land cover. Export your findings.')
a,b,c = st.columns(3)
a.metric('Land-cover classes', '10')
b.metric('Scenes analyzed', len(st.session_state.results))
c.metric('Model', 'Pretrained CNN')
st.divider()

if section == 'Scene analysis':
    files = st.file_uploader('Add satellite scenes', type=['jpg','jpeg','png','tif','tiff'], accept_multiple_files=True,
                             help='RGB satellite images work best. Up to 20 scenes per batch; 20 MB per file.')
    sample = st.selectbox('Or explore an included scene', ['Choose a sample'] + [p.name for p in sorted(ROOT.glob('test*.jpg'))])
    scenes = [(f.name, f.getvalue()) for f in files] if files else ([(sample, (ROOT/sample).read_bytes())] if sample != 'Choose a sample' else [])
    if scenes:
        selected = st.selectbox('Active scene', range(len(scenes)), format_func=lambda i: scenes[i][0])
        name, data = scenes[selected]
        left,right = st.columns([1.7,1])
        with left:
            st.subheader('Scene viewport')
            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
                zoom = st.slider('Viewport zoom', 1, 4, 1)
                if zoom > 1:
                    x = st.slider('Horizontal center', 0, 100, 50)/100
                    y = st.slider('Vertical center', 0, 100, 50)/100
                    w,h = image.size; cw,ch = w/zoom,h/zoom
                    x0=(w-cw)*x; y0=(h-ch)*y
                    view=image.crop((x0,y0,x0+cw,y0+ch))
                else: view=image
                st.image(view, width='stretch')
                st.caption(f'{name} · {image.width} × {image.height} pixels · RGB | Zoom changes the viewport only; inference uses the full scene.')
            except Exception:
                st.error('This file could not be displayed. Choose a valid supported image.')
        with right:
            st.subheader('Analysis controls')
            st.write('Run the original trained classifier on your selected imagery.')
            run_one = st.button('Analyze active scene', type='primary', width='stretch')
            run_all = st.button('Analyze entire batch', width='stretch')
            if run_one or run_all:
                targets = scenes if run_all else [(name,data)]
                if len(targets)>20:
                    st.error('Select no more than 20 scenes per batch.')
                else:
                    progress=st.progress(0)
                    for i,(filename,blob) in enumerate(targets):
                        try:
                            if len(blob)>20*1024*1024: raise ValueError('File exceeds 20 MB.')
                            with st.spinner('Classifying '+filename):
                                result=classify(blob,Path(filename).suffix.lower())
                            result.update(filename=filename, sha256=hashlib.sha256(blob).hexdigest(), timestamp=datetime.now(timezone.utc).isoformat())
                            st.session_state.results.append(result)
                        except Exception as exc:
                            st.error(f'{filename}: {exc}')
                        progress.progress((i+1)/len(targets))
            matches=[r for r in st.session_state.results if r['sha256']==hashlib.sha256(data).hexdigest()]
            if matches:
                result=matches[-1]
                st.metric(result['predicted_class'], f"{result['confidence']:.1f}%")
                if result['confidence']<threshold: st.warning('Review recommended: confidence is below your threshold.')
                else: st.success('Analysis complete')
                st.bar_chart(pd.DataFrame(result['all_probabilities'].items(),columns=['Class','Probability (%)']).set_index('Class'), horizontal=True)
                st.download_button('Download scene report', json.dumps(result,indent=2), 'scene-report.json','application/json')
            else: st.info('Ready for analysis. Results will appear here.')
    else:
        st.info('Upload satellite imagery or choose an included sample to begin.')
        cols=st.columns(4)
        for col,p in zip(cols,sorted(ROOT.glob('test*.jpg'))):
            col.image(str(p),caption=p.name,width='stretch')
elif section == 'Mission history':
    if st.session_state.results:
        rows=[{k:v for k,v in r.items() if k!='all_probabilities'} for r in st.session_state.results]
        frame=pd.DataFrame(rows)
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('Export CSV',frame.to_csv(index=False),'terrascope-results.csv','text/csv')
        st.download_button('Export complete JSON',json.dumps(st.session_state.results,indent=2),'terrascope-results.json','application/json')
        st.bar_chart(frame['predicted_class'].value_counts())
    else: st.info('Analyze a scene to start your mission history. History is held only in this browser session.')
else:
    st.subheader('Original training evidence')
    st.caption('These artifacts were supplied with the project. They have not been regenerated or independently re-evaluated.')
    st.warning('The supplied classification report records 63.93% overall accuracy. Prediction confidence is not a guarantee of correctness.')
    for filename in ['confusion_matrix.png','training_history.png']:
        p=ROOT/'outputs/plots'/filename
        if p.exists(): st.image(str(p),caption=filename,width='stretch')
    p=ROOT/'outputs/plots/classification_report.txt'
    if p.exists(): st.text(p.read_text())
    st.code(hashlib.sha256((ROOT/'models/satellite_cnn_model.h5').read_bytes()).hexdigest(),language=None)
    st.caption('SHA-256 fingerprint of the original trained model.')
