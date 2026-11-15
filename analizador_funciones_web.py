import math
import warnings
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.optimize import brentq
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

st.set_page_config(page_title='Atlas | Analizador de funciones', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
X = sp.Symbol('x', real=True)
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
ALLOWED = {'x': X, 'sin':sp.sin, 'cos':sp.cos, 'tan':sp.tan, 'asin':sp.asin, 'acos':sp.acos, 'atan':sp.atan,
           'exp':sp.exp, 'log':sp.log, 'ln':sp.log, 'sqrt':sp.sqrt, 'Abs':sp.Abs, 'abs':sp.Abs,
           'pi':sp.pi, 'E':sp.E, 'sinh':sp.sinh, 'cosh':sp.cosh, 'tanh':sp.tanh, 'sign':sp.sign,
           'floor':sp.floor, 'ceiling':sp.ceiling, 'Piecewise':sp.Piecewise}
COL = {'f':'#3b82f6','up':'#16a34a','down':'#ef4444','max':'#d97706','min':'#8b5cf6',
       'infl':'#06b6d4','asym':'#94a3b8','concup':'#0d9488','concdown':'#e879f9'}


st.markdown("""
<style>
/* Diseño general */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background: #101b2c;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* Campo de entrada de funciones */
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #111827 !important;
    font-size: 19px !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    caret-color: #111827 !important;
}

[data-testid="stSidebar"] input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Desplegable de ejemplos */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #111827 !important;
    font-weight: 600 !important;
}

/* Recuadro para copiar expresiones */
[data-testid="stSidebar"] pre {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
}

[data-testid="stSidebar"] pre,
[data-testid="stSidebar"] pre code,
[data-testid="stSidebar"] pre code span {
    color: #111827 !important;
    font-size: 16px !important;
}

/* Encabezado ATLAS */
.hero {
    background: linear-gradient(115deg, #10243c, #163a5c);
    padding: 32px 27px 21px 27px;
    border-radius: 18px;
    color: white;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 2rem;
    letter-spacing: -.04em;
    margin: 0;
    color: #fff;
}

.hero p {
    color: #cbd5e1;
    margin: 6px 0 0;
}

.kicker {
    letter-spacing: .17em;
    font-size: .68rem;
    color: #7dd3fc;
    font-weight: 700;
}

/* Indicadores matemáticos */
[data-testid="stMetric"] {
    background: rgba(148,163,184,.08);
    border: 1px solid rgba(148,163,184,.17);
    padding: 12px;
    border-radius: 13px;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p {
    color: #111827 !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="kicker">LABORATORIO DE MATEMÁTICA · PYTHON + SYMPY</div><h1>ATLAS · Analizador de funciones reales</h1></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header('01 · Función')
    expr_input = st.text_input('Introduce f(x)', value='(x^3*(x-5)/((x-5)*(x^2-4)))', help='Usa x, ^, sqrt(...), abs(...), sin(...), log(...).')
    examples={'Cúbica':'x^3 - 3*x','Racional':'(x^2 + 1)/(x - 2)','Valor absoluto':'abs(x)','Raíz':'sqrt(x+2)','Logaritmo':'log(x)','Trigonométrica':'sin(x)','Oscilatoria':'sin(x)/x','Cúspide':'abs(x)^(2/3)'}
    chosen=st.selectbox('Ejemplos para copiar', ['—']+list(examples))
    if chosen!='—':
        st.caption('Copia esta expresión en el campo superior:')
        st.code(examples[chosen],language=None)
        st.divider()
    st.header('02 · Dominio de análisis')
    tipo_dominio = st.radio(
        'Selecciona el dominio:',
        ['x ∈ ℝ', 'x ∈ I'],
        horizontal=True
    )

    if tipo_dominio == 'x ∈ I':

        if 'limite_inferior' not in st.session_state:
            st.session_state.limite_inferior = '-4'

        if 'limite_superior' not in st.session_state:
            st.session_state.limite_superior = '6'

        if 'extremo_izq' not in st.session_state:
            st.session_state.extremo_izq = '('

        if 'extremo_der' not in st.session_state:
            st.session_state.extremo_der = ']'

        def cambiar_izquierdo():
            st.session_state.extremo_izq = (
                '[' if st.session_state.extremo_izq == '(' else '('
            )

        def cambiar_derecho():
            st.session_state.extremo_der = (
                ')' if st.session_state.extremo_der == ']' else ']'
            )

        c1, c2, c3, c4, c5 = st.columns(
            [1.2, 2, 0.4, 2, 1.2],
            vertical_alignment="center"
        )

        with c1:
            st.button(
                st.session_state.extremo_izq,
                key='boton_izq',
                on_click=cambiar_izquierdo
            )
            extremo_izq = st.session_state.extremo_izq

        with c2:
            a_dominio = st.text_input(
                'Inferior',
                key='limite_inferior',
                label_visibility='collapsed'
            )

        with c3:
            st.markdown('### ,')

        with c4:
            b_dominio = st.text_input(
                'Superior',
                key='limite_superior',
                label_visibility='collapsed'
            )

        with c5:
            st.button(
                st.session_state.extremo_der,
                key='boton_der',
                on_click=cambiar_derecho
            )
            extremo_der = st.session_state.extremo_der

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] hr {
            margin-top: 0.3rem;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.header('03 · Elementos visibles')
    show_asym=st.checkbox('Asíntotas',False)
    show_holes=st.checkbox('Discontinuidades removibles (huecos)',False)
    show_crit=st.checkbox('Puntos críticos',False)
    show_ext=st.checkbox('Máximos y mínimos',False)
    show_mono=st.checkbox('Colorear crecimiento / decrecimiento',False)
    show_infl=st.checkbox('Puntos de inflexión',False)
    show_conc=st.checkbox('Colorear concavidad',False)
    show_tangent = st.checkbox('Recta tangente', False)
    show_legend=st.checkbox('Leyenda',True)
    x0_tangent = st.number_input('Punto de tangencia (x₀)', value=1.0, step=0.5) if show_tangent else None
    st.header('02 · Ventana gráfica')
    xmin,xmax=st.slider('Intervalo visible de x',-30.0,30.0,(-7.0,7.0),0.5)
    samples=st.slider('Resolución de muestreo',500,5000,2000,500)
    st.divider()
    st.caption('Los controles solo cambian la visualización, no la función.')

def latex_conjunto(conjunto):
    if conjunto == sp.S.Reals:
        return r'\mathbb{R}'

    if conjunto == sp.S.EmptySet:
        return r'\varnothing'
    try:
        excluidos = sp.S.Reals - conjunto

        if isinstance(excluidos, sp.FiniteSet):
            elementos = sorted(
                excluidos,
                key=lambda v: float(sp.N(v))
            )

            valores = ', '.join(
                sp.latex(v) for v in elementos
            )

            return r'\mathbb{R}\setminus\{' + valores + r'\}'

    except (TypeError, ValueError, NotImplementedError):
        pass

    return sp.latex(conjunto)
def parse_safe(s):
    if len(s)>220 or '__' in s or any(t in s for t in ('[',']','{','}',';',':','\\','"',"'",'.')):
        raise ValueError('Expresión no admitida. Utiliza únicamente funciones matemáticas de la lista.')
    import re
    tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*',s)
    unknown=[t for t in tokens if t not in ALLOWED]
    if unknown: raise ValueError('Símbolos no admitidos: '+', '.join(sorted(set(unknown))))
    # SymPy parser is not a security boundary: this app is intended for trusted/local classroom use.
    e=parse_expr(s,local_dict=ALLOWED,global_dict={'__builtins__':{},'Integer':sp.Integer,'Float':sp.Float,'Rational':sp.Rational,'Symbol':sp.Symbol,'Add':sp.Add,'Mul':sp.Mul,'Pow':sp.Pow,'Function':sp.Function},transformations=TRANSFORMS,evaluate=False)
    if e.free_symbols-{X}:raise ValueError('La función debe depender únicamente de x.')
    return e


def real_float(v):
    try:
        n=complex(sp.N(v,14))
        if abs(n.imag)>1e-8 or not math.isfinite(n.real):return None
        return float(n.real)
    except Exception:return None


def finite_set_points(sol,lo,hi,cap=120):
    if isinstance(sol,sp.FiniteSet):
        result=[]
        for v in sol:
            z=real_float(v)
            if z is not None and lo-1e-8<=z<=hi+1e-8:result.append((z,v))
        return sorted(result,key=lambda t:t[0])[:cap],True
    if isinstance(sol,sp.Union):
        accum=[]; exact=True
        for a in sol.args:
            p,ok=finite_set_points(a,lo,hi,cap);accum.extend(p);exact &= ok
        return sorted(accum,key=lambda t:t[0])[:cap],exact
    return [],False


def evaluate_numeric(expr,xvals):
    fn=sp.lambdify(X,expr,modules=['numpy'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(all='ignore'):
            vals=fn(xvals)
    arr=np.asarray(vals)
    if arr.ndim==0:arr=np.full_like(xvals,arr,dtype=complex)
    if arr.shape!=xvals.shape:arr=np.broadcast_to(arr,xvals.shape)
    if np.iscomplexobj(arr):
        valid=np.abs(np.imag(arr))<1e-8
        arr=np.where(valid,np.real(arr),np.nan)
    return np.asarray(arr,dtype=float)


def dedup(items,eps=1e-5):
    out=[]
    for x,origin in sorted(items,key=lambda t:t[0]):
        if not out or abs(x-out[-1][0])>eps:out.append((x,origin))
        elif origin=='exact':out[-1]=(x,origin)
    return out


def numeric_roots(expr,lo,hi,n=1100):
    xs=np.linspace(lo,hi,n)
    try:ys=evaluate_numeric(expr,xs)
    except Exception:return []
    roots=[]
    for i in range(len(xs)-1):
        a,b=ys[i:i+2]
        if not (np.isfinite(a) and np.isfinite(b)):continue
        if abs(a)<1e-7:roots.append((float(xs[i]),'approx'))
        if a*b<0:
            try:
                fn=sp.lambdify(X,expr,modules='numpy')
                root=brentq(fn,float(xs[i]),float(xs[i+1]),xtol=1e-11)
                # reject apparent roots across poles/discontinuities
                rr=abs(float(fn(root)))
                if rr<1e-5:roots.append((root,'approx'))
            except Exception:pass
    return dedup(roots)


def domain_contains(dom,pt):
    try:return dom.contains(sp.Float(pt))==sp.S.true
    except Exception:return False


def sign_near(d,at,side,lo,hi):
    span=max(hi-lo,1.0)
    for k in (1e-4,1e-3,1e-2,5e-2):
        delta=span*k
        v=at+side*delta
        if not lo<=v<=hi:continue
        try:
            z=real_float(d.subs(X,v))
            if z is not None and abs(z)>1e-8:return 1 if z>0 else -1
        except Exception:pass
    return 0


def asymptotes(expr,lo,hi):
    lines=[]
    den=sp.denom(sp.together(expr))
    if den!=1:
        try:
            poles,_=finite_set_points(sp.solveset(den,X,domain=sp.S.Reals),lo,hi)
            for xx,ex in poles:
                if sp.simplify(expr).subs(X,ex) in (sp.zoo,sp.oo,-sp.oo,sp.nan):
                    lines.append(('vertical',xx,f'x = {sp.latex(ex)}'))
                else:
                    try:
                        l1=sp.limit(expr,X,ex,dir='-');l2=sp.limit(expr,X,ex,dir='+')
                        if any(t in (sp.oo,-sp.oo) for t in (l1,l2)):
                            lines.append(('vertical',xx,f'x = {sp.latex(ex)}'))
                    except Exception:pass
        except Exception:pass
    for end in (sp.oo,-sp.oo):
        try:
            m=sp.limit(expr/X,X,end)
            if not m.is_real or not m.is_finite:continue
            b=sp.limit(expr-m*X,X,end)
            if not b.is_real or not b.is_finite:continue
            mm=real_float(m);bb=real_float(b)
            if mm is None or bb is None:continue
            if any(t[0]=='line' and abs(t[1][0]-mm)<1e-8 and abs(t[1][1]-bb)<1e-8 for t in lines):continue
            lines.append(('line',(mm,bb),sp.latex(sp.Eq(sp.Symbol('y'),m*X+b))))
        except Exception:pass
    return lines

try:f=parse_safe(expr_input)
except Exception as e:
    st.error(f'No se pudo interpretar la función: {e}')
    st.stop()

with st.spinner('Analizando la función…'):
    f_calculo = sp.cancel(f)
    try:df=sp.diff(f_calculo,X); d2=sp.diff(df,X)
    except Exception as e:st.error(f'No se pudo derivar esta función: {e}');st.stop()
    try:dom=continuous_domain(f,X,sp.S.Reals);dom_ok=True
    except Exception:dom=sp.S.Reals;dom_ok=False
    try:
        if tipo_dominio == 'x ∈ I':
            a_rng = float(a_dominio)
            b_rng = float(b_dominio)

            if not (np.isfinite(a_rng) and np.isfinite(b_rng) and a_rng < b_rng):
                st.error('El intervalo debe tener extremos finitos y cumplir a < b.')
                st.stop()

            intervalo = sp.Interval(
                a_rng,
                b_rng,
                left_open=(extremo_izq == '('),
                right_open=(extremo_der == ')')
            )        

            if not (np.isfinite(a_rng) and np.isfinite(b_rng) and a_rng < b_rng):
                st.error('El intervalo debe tener extremos finitos y cumplir a < b.')
                st.stop()

            intervalo = sp.Interval(
                a_rng, b_rng,
                left_open=(extremo_izq == '('),
                right_open=(extremo_der == ')')
            )

            dominio_analisis = dom.intersect(intervalo)

        else:
            dominio_analisis = dom

    except (ValueError, TypeError) as e:
        st.error(f'Intervalo de análisis no válido: {e}')
        st.stop()

    rng = function_range(f_calculo, X, dominio_analisis)
    try:
        sol=sp.solveset(df,X,domain=sp.S.Reals)
        crit,crit_complete=finite_set_points(sol,xmin,xmax)
    except Exception:crit=[];crit_complete=False
    if not crit_complete:
        crit=dedup(crit+numeric_roots(df,xmin,xmax))
    # Include nondifferentiable candidates found via derivative singularities, within domain.
    try:
        from sympy.calculus.singularities import singularities
        nondiff,_=finite_set_points(singularities(df,X),xmin,xmax)
        for p,_ in nondiff:
            if domain_contains(dom,p):crit.append((p,'nondiff'))
    except Exception:pass
    crit=dedup(crit)
    if tipo_dominio == 'x ∈ I':
        crit = [
            (p, origen)
            for p, origen in crit
            if dominio_analisis.contains(sp.Float(p)) == sp.S.true
        ]
    # Confirm that f is finite at each candidate and classify using signs on both sides.
    extrema=[];other=[]
    for p,origin in crit:
        fv=real_float(f.subs(X,p))
        if fv is None:continue
        left=sign_near(df,p,-1,xmin,xmax);right=sign_near(df,p,1,xmin,xmax)
        if left>0 and right<0:extrema.append((p,fv,'Máximo',origin))
        elif left<0 and right>0:extrema.append((p,fv,'Mínimo',origin))
        else:other.append((p,fv,origin))
    # Extremos relativos unilaterales en los bordes cerrados del intervalo.
    if tipo_dominio == 'x ∈ I':
        bordes = [
            (a_rng, extremo_izq == '[', 1),
            (b_rng, extremo_der == ']', -1)
        ]

        for p, cerrado, lado in bordes:
            if not cerrado:
                continue

            if not domain_contains(dom, p):
                continue

            fv = real_float(f.subs(X, p))
            if fv is None:
                continue

            signo = sign_near(df, p, lado, a_rng, b_rng)

            if signo == 0:
                continue

            if lado == 1:
                tipo = 'Mínimo' if signo > 0 else 'Máximo'
            else:
                tipo = 'Máximo' if signo > 0 else 'Mínimo'

            extrema.append((p, fv, tipo, 'extremo_intervalo'))
    try:
        infsol=sp.solveset(d2,X,domain=sp.S.Reals)
        infl_candidates,inf_complete=finite_set_points(infsol,xmin,xmax)
    except Exception:infl_candidates=[];inf_complete=False
    if not inf_complete:infl_candidates=dedup(infl_candidates+numeric_roots(d2,xmin,xmax))
    try:
        from sympy.calculus.singularities import singularities
        pts,_=finite_set_points(singularities(d2,X),xmin,xmax)
        infl_candidates=dedup(infl_candidates+[(p,'nondiff') for p,_ in pts])
    except Exception:pass
    if tipo_dominio == 'x ∈ I':
        infl_candidates = [
            (p, origen)
            for p, origen in infl_candidates
            if dominio_analisis.contains(sp.Float(p)) == sp.S.true
        ]
    infl=[]
    for p,origin in infl_candidates:
        fv=real_float(f.subs(X,p))
        if fv is None:continue
        left=sign_near(d2,p,-1,xmin,xmax);right=sign_near(d2,p,1,xmin,xmax)
        if left*right<0:infl.append((p,fv,origin))
    # Discontinuidades removibles: conservar los factores del denominador original.
    # Solo se clasifican como huecos cuando el limite bilateral es finito y real.
    holes=[]
    try:
        original_den=sp.fraction(f,exact=True)[1]
        excluded=sp.solveset(original_den,X,domain=sp.S.Reals)
        den_points,_=finite_set_points(excluded,xmin,xmax)
        for xx,ex in den_points:
            try:
                if dom.contains(ex) != sp.S.false:
                    continue
                lv=sp.limit(f,X,ex,dir='-')
                rv=sp.limit(f,X,ex,dir='+')
                if sp.simplify(lv-rv)==0:
                    yy=real_float(lv)
                    if yy is not None: holes.append((xx,yy,ex))
            except Exception:
                pass
    except Exception:
        pass
    if tipo_dominio == 'x ∈ I':
        holes = [
            (hx, hy, ex)
            for hx, hy, ex in holes
            if intervalo.contains(ex) == sp.S.true
        ]
        
    asym=asymptotes(f_calculo,xmin,xmax)
    if tipo_dominio == 'x ∈ I':
        asym = [
            (kind, value, label)
            for kind, value, label in asym
            if kind == 'vertical'
            and intervalo.contains(sp.Float(value)) == sp.S.true
        ]



left,right=st.columns([3.25,1.25],gap='large')
with left:
    if tipo_dominio == 'x ∈ I':
        a = a_rng
        b = b_rng
        xs = np.linspace(a, b, samples)
    else:
        xs = np.linspace(xmin, xmax, samples)
    try:ys=evaluate_numeric(f,xs); dys=evaluate_numeric(df,xs);d2ys=evaluate_numeric(d2,xs)
    except Exception as e:st.error(f'No se pudo representar numéricamente la función: {e}');st.stop()
    ys[~np.isfinite(ys)]=np.nan
    # Prevent drawing across discontinuities and asymptotes.
    yr=ys[np.isfinite(ys)]
    scale=np.nanpercentile(np.abs(yr),75) if len(yr) else 1.0
    threshold=max(15.0,scale*10)
    ys[np.abs(ys)>threshold]=np.nan
    for kind,value,_ in asym:
        if kind=='vertical':ys[np.abs(xs-value)<(xmax-xmin)/samples*2]=np.nan
    for hx,hy,_ in holes:
        ys[np.abs(xs-hx)<(xmax-xmin)/samples*1.5]=np.nan
    jumps=np.where(np.abs(np.diff(ys))>max(8,scale*5))[0]
    ys[jumps+1]=np.nan
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name='f(x)',line=dict(color=COL['f'],width=3.4),connectgaps=False,hovertemplate='x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>'))
    if show_tangent:
        x0 = float(x0_tangent)

        if not domain_contains(dom, x0):
            st.warning('El punto seleccionado no pertenece al dominio de la función.')
        else:
            y0 = real_float(f.subs(X, x0))
            m = real_float(df.subs(X, x0))

            if y0 is None or m is None or not np.isfinite(y0) or not np.isfinite(m):
                st.warning('No existe una recta tangente ordinaria en el punto seleccionado.')
            else:
                xt = np.array([xmin, xmax])
                yt = y0 + m * (xt - x0)
                b_tangent = y0 - m * x0

                m_exacto = sp.simplify(df.subs(X, sp.Rational(str(x0_tangent))))
                b_exacto = sp.simplify(
                    f.subs(X, sp.Rational(str(x0_tangent)))
                    - m_exacto * sp.Rational(str(x0_tangent))
                )

                def formato_fraccion(valor):
                    valor = sp.Rational(valor)
                    sup = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
                    sub = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

                    if valor.q == 1:
                        return str(valor.p)

                    return (
                        str(valor.p).translate(sup)
                        + '⁄'
                        + str(valor.q).translate(sub)
                    )

                if m_exacto == 0:
                    etiqueta_tangent = 'y = ' + formato_fraccion(b_exacto)
                else:
                    etiqueta_tangent = 'y = ' + formato_fraccion(m_exacto) + 'x'

                    if b_exacto != 0:
                        etiqueta_tangent += (
                            (' + ' if b_exacto > 0 else ' − ')
                            + formato_fraccion(abs(b_exacto))
                    )

                fig.add_trace(go.Scatter(
                    x=xt,
                    y=yt,
                    mode='lines',
                    name=etiqueta_tangent,
                    line=dict(color='#dc2626', width=2.5, dash='solid')
                ))

                fig.add_trace(go.Scatter(
                    x=[x0],
                    y=[y0],
                    mode='markers',
                    name='Punto de tangencia',
                    marker=dict(color='#dc2626', size=11)
                ))
                fig.add_annotation(
                    x=x0, y=y0,
                    text=etiqueta_tangent,
                    showarrow=False,
                    xshift=75,
                    yshift=35,
                    font=dict(size=16, color='#dc2626'),
                    bgcolor='rgba(255,255,255,0.85)'
                )
    if tipo_dominio == 'x ∈ I':
        for x_ext, simbolo in [(a, extremo_izq), (b, extremo_der)]:
            if not domain_contains(dom, x_ext):
                continue

            y_ext = real_float(f.subs(X, x_ext))
            if y_ext is None:
                continue

            cerrado = simbolo in ('[', ']')

            fig.add_trace(go.Scatter(
                x=[x_ext],
                y=[y_ext],
                mode='markers',
                marker=dict(
                    symbol='circle' if cerrado else 'circle-open',
                    size=14,
                    color='#2563eb',
                    line=dict(color='#2563eb', width=3)
                ),
                name='Extremo cerrado' if cerrado else 'Extremo abierto',
                hovertemplate=f'Extremo: ({x_ext:g}, {y_ext:.4g})<extra></extra>'
            ))
    if show_mono or show_conc:
        vals=dys if show_mono else d2ys
        vals=np.where(np.isfinite(vals),vals,0)
        for positive,label,color in [(True,'Creciente' if show_mono else 'Cóncava hacia arriba',COL['up'] if show_mono else COL['concup']),
                                     (False,'Decreciente' if show_mono else 'Cóncava hacia abajo',COL['down'] if show_mono else COL['concdown'])]:
            mask=vals>1e-8 if positive else vals< -1e-8
            segment=np.where(mask,ys,np.nan)
            fig.add_trace(go.Scatter(x=xs,y=segment,mode='lines',name=label,line=dict(color=color,width=4.1),connectgaps=False,hoverinfo='skip'))
    if show_asym:
        for kind,value,label in asym:
            if kind=='vertical':
                fig.add_vline(x=value,line_color=COL['asym'],line_dash='dash',line_width=1.5)
                fig.add_annotation(x=value,y=0.85,yref='paper',text=label,showarrow=False,font=dict(color=COL['asym'],size=15),bgcolor='rgba(255,255,255,0.95)',bordercolor='#94a3b8',borderwidth=1,borderpad=5)          
            else:
                m,b=value
                m_exacto = sp.nsimplify(m, rational=True)
                b_exacto = sp.nsimplify(b, rational=True)

                expresion_asintota = sp.expand(m_exacto * X + b_exacto)
                etiqueta_asintota = "y = " + sp.sstr(expresion_asintota)

                fig.add_trace(go.Scatter(
                    x=[xmin,xmax],
                    y=[m*xmin+b,m*xmax+b],
                    mode='lines',
                    name='Asíntota: '+label,
                    line=dict(
                        color=COL['asym'],
                        dash='dash',
                        width=1.5
                    )
                ))

                # Posiciones candidatas sobre la asíntota
                candidatos = np.linspace(
                    xmin + 0.12*(xmax-xmin),
                    xmax - 0.12*(xmax-xmin),
                    25
                )

                puntos_evitar = [
                    (p[0], p[1]) for p in extrema
                ]

                # Elegir la posición más alejada de los extremos relativos
                if puntos_evitar:
                    escala_x = max(xmax-xmin, 1)
                    escala_y = max(float(np.nanmax(ys) - np.nanmin(ys)), 1)

                    x_etiqueta = max(
                        candidatos,
                        key=lambda x: min(
                            ((x-px)/escala_x)**2
                            + ((m*x+b-py)/escala_y)**2
                            for px, py in puntos_evitar
                        )
                    )
                else:
                    x_etiqueta = xmin + 0.75*(xmax-xmin)

                y_etiqueta = m*x_etiqueta+b

                fig.add_annotation(
                    x=x_etiqueta,
                    y=y_etiqueta,
                    text=etiqueta_asintota,
                    showarrow=False,
                    xshift=0,
                    yshift=18,
                    font=dict(
                        color=COL['asym'],
                        size=15
                    ),
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#94a3b8',
                    borderwidth=1,
                    borderpad=5
                    )

    if show_holes and holes:
        fig.add_trace(go.Scatter(x=[h[0] for h in holes],y=[h[1] for h in holes],
            mode='markers+text',name='Discontinuidad removible',
            marker=dict(symbol='circle-open',size=15,color='#fbbf24',line=dict(width=3)),
            text=[f'Hueco ({h[0]:.4g}, {h[1]:.4g})' for h in holes],
            textposition='top right',textfont=dict(color='#fbbf24',size=24)))
    if show_ext:
        for kind,color in [('Máximo',COL['max']),('Mínimo',COL['min'])]:
            pts=[a for a in extrema if a[2]==kind]
            if pts:
                fig.add_trace(go.Scatter(x=[p[0] for p in pts],y=[p[1] for p in pts],mode='markers+text',name=kind+' relativo',marker=dict(size=12,color=color,line=dict(color='white',width=1.5)),text=[f'{kind} ({p[0]:.3g}, {p[1]:.3g})' for p in pts],textposition='top center',textfont=dict(color=color,size=24)))
    puntos_criticos = list(other)

    if not show_ext:
        puntos_criticos += [
            (p, fv, origin)
            for p, fv, kind, origin in extrema
        ]
    if show_crit and puntos_criticos:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in puntos_criticos],
            y=[p[1] for p in puntos_criticos],
            mode='markers+text',
            name='Puntos criticos',
            marker=dict(
                size=14,
                symbol='diamond',
                color='#f59e0b'
            ),
            text=['Crítico' for _ in puntos_criticos],
            textposition='top right',
            textfont=dict(
                size=18,
                color='#b45309'
            )
        ))
    if show_infl and infl:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in infl],
            y=[p[1] for p in infl],
            mode='markers+text',
            name='Puntos de inflexión',
            marker=dict(
                size=11,
                symbol='circle',
                color=COL['infl'],
                line=dict(color='white', width=1.5)
            ),
            text=[
                f'Inflexión ({p[0]:.3g}, {p[1]:.3g})'
                for p in infl
            ],
            textposition='bottom right',
            textfont=dict(
                color=COL['infl'],
                size=24
            )
        ))


    
    # Obtener los límites verticales visibles de la gráfica
    valores_y = []
    for traza in fig.data:
        if traza.y is not None:
            try:
                datos = np.asarray(traza.y, dtype=float)
                datos = datos[np.isfinite(datos)]
                valores_y.extend(datos.tolist())
            except (TypeError, ValueError):
                pass

    if valores_y:
        ymin = float(np.percentile(valores_y, 2))
        ymax = float(np.percentile(valores_y, 98))
    else:
        ymin, ymax = -10.0, 10.0

    # Garantizar que el origen sea visible
    ymin = min(ymin, 0.0)
    ymax = max(ymax, 0.0)
    margen = max((ymax - ymin) * 0.08, 1.0)
    ymin -= margen
    ymax += margen

    fig.update_layout(
        height=650,
        margin=dict(l=15, r=25, t=25, b=25),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1f2937', size=13),
        hovermode='closest',
        showlegend=show_legend,
        legend=dict(orientation='h', y=-0.12, x=0),
        xaxis=dict(
            range=[xmin, xmax],
            showgrid=False,
            showticklabels=False,
            ticks='',
            showline=False,
            zeroline=False,
            title=None
        ),
        yaxis=dict(
            range=[ymin, ymax],
            showgrid=False,
            showticklabels=False,
            ticks='',
            showline=False,
            zeroline=False,
            title=None
        ),
        uirevision='atlas',
        dragmode='pan'
    )

    # Ejes cartesianos con flechas
    fig.add_annotation(
        x=xmax, y=0, ax=xmin, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor='#1f2937'
    )

    fig.add_annotation(
        x=0, y=ymax, ax=0, ay=ymin,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor='#1f2937'
    )

    # Graduaciones numéricas sobre el eje X
    for k in range(math.ceil(xmin), math.floor(xmax) + 1):
        fig.add_shape(
            type='line',
            x0=k, x1=k,
            y0=-0.008*(ymax-ymin),
            y1=0.008*(ymax-ymin),
            line=dict(color='#334155', width=1)
        )
        fig.add_annotation(
            x=k, y=0,
            text=str(k),
            showarrow=False,
            yshift=-17,
            font=dict(size=11, color='#334155')
        )

    # Graduaciones numéricas sobre el eje Y
    paso_y = max(1, math.ceil((ymax-ymin)/10))

    for k in range(
        math.ceil(ymin/paso_y)*paso_y,
        math.floor(ymax/paso_y)*paso_y + 1,
        paso_y
    ):
        if k == 0:
            continue

        fig.add_shape(
            type='line',
            x0=-0.008*(xmax-xmin),
            x1=0.008*(xmax-xmin),
            y0=k, y1=k,
            line=dict(color='#334155', width=1)
        )
        fig.add_annotation(
            x=0, y=k,
            text=str(k),
            showarrow=False,
            xshift=-17,
            font=dict(size=11, color='#334155')
        )

    # Identificación de los ejes
    fig.add_annotation(
        x=xmax, y=0, text='x',
        showarrow=False,
        xshift=-12, yshift=18,
        font=dict(size=18, color='#111827')
    )

    fig.add_annotation(
        x=0, y=ymax, text='y',
        showarrow=False,
        xshift=18, yshift=-15,
        font=dict(size=18, color='#111827')
    )
    
    # ATLAS: etiquetas de alta visibilidad para el aula
    fig.update_annotations(
        font=dict(size=17, color='#1e293b')
    )

    # Conservar los colores propios de las etiquetas matemáticas
    for anotacion in fig.layout.annotations:
        if anotacion.text is None:
            continue

        texto = str(anotacion.text).lower()

        if 'x =' in texto or 'y =' in texto:
            anotacion.font = dict(
                size=18,
                color='#475569'
            )
        elif texto.strip() in ('x', 'y'):
            anotacion.font = dict(
                size=22,
                color='#0f172a'
            )
        elif anotacion.text.strip().lstrip('-').isdigit():
            anotacion.font = dict(
                size=15,
                color='#334155'
            )

    # Etiquetas de puntos especiales
    for traza in fig.data:
        nombre = str(traza.name or '').lower()

        if traza.mode and 'text' in traza.mode:
            if 'máximo' in nombre or 'mínimo' in nombre:
                traza.textfont = dict(size=18, color='#9a3412' if 'máximo' in nombre else '#6d28d9')
                traza.marker.size = 14
            elif 'inflexión' in nombre:
                traza.textfont = dict(size=18, color='#0891b2')
                traza.marker.size = 24
            elif 'discontinuidad' in nombre:
                traza.textfont = dict(size=18, color='#b45309')
                traza.marker.size = 17
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displaylogo': False,
            'scrollZoom': False,
            'doubleClick': 'reset',
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'atlas_funcion',
                'scale': 3
            }
        }
    )
    st.caption('Interactúa: zoom, desplazamiento, lectura de coordenadas y exportación PNG desde la barra de la gráfica. Los resultados numéricos dependen de la ventana visible.')
with right:
    st.subheader('Ficha matemática')
    st.latex('f(x)='+sp.latex(f))
    if holes:
        st.caption('Se conserva el dominio de la expresión introducida, incluso si hay factores cancelables.')
    with st.expander('Primera derivada', expanded=False):
        df_presentacion = sp.factor(sp.cancel(df))
        st.latex("f'(x)=" + sp.latex(df_presentacion))

    with st.expander('Segunda derivada', expanded=False):
        d2_presentacion = sp.factor(sp.cancel(d2))
        st.latex("f''(x)=" + sp.latex(d2_presentacion))
    if tipo_dominio == 'x ∈ I':
        st.markdown('**Dominio de análisis**')
        if dom_ok:
            st.latex(r'D_{f,I} = ' + latex_conjunto(dominio_analisis))
        else:
            st.latex(r'D_{f,I} = \text{No determinado}')
    else:
        st.markdown('**Dominio real**')
        if dom_ok:
            st.latex(r'D_f = ' + latex_conjunto(dom))
        else:
            st.latex(r'D_f = \text{No determinado}')

    st.markdown('**Rango**')
    if rng is not None:
        st.latex(r'\operatorname{Im}(f) = ' + latex_conjunto(rng))
    else:
        st.latex(r'\operatorname{Im}(f) = \text{No determinado simbólicamente}')

with st.expander('Análisis de signos y metodología',expanded=False):
    st.markdown('**Criterio de la primera derivada:** máximo relativo si f\' cambia de + a −; mínimo si cambia de − a +. Se comprueban ambos lados de cada candidato detectado.')
    if crit:
        rows=[]
        for p,origin in crit:
            rows.append({'x':round(p,7),'f\'(izquierda)':sign_near(df,p,-1,xmin,xmax),'f\'(derecha)':sign_near(df,p,1,xmin,xmax),'Origen':origin})
        st.dataframe(rows,use_container_width=True,hide_index=True)
    else:st.info('No se encontraron candidatos críticos en el intervalo visible.')
    st.warning('Alcance: el motor combina SymPy con búsqueda numérica acotada. No garantiza hallar todas las raíces, discontinuidades, asíntotas ni puntos críticos de una función arbitraria. La clasificación por signos es local y numérica; para funciones muy oscilatorias o singularidades próximas requiere verificación matemática. La ventana gráfica no sustituye el dominio global.')

st.divider()
st.caption('ATLAS · Prototipo 1.1 | Python · SymPy · NumPy · SciPy · Plotly · Streamlit | Uso didáctico universitario')
