import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { AgGridReact } from 'ag-grid-react';
import {
  themeQuartz,
  colorSchemeDarkBlue,
  AllCommunityModule,
  ModuleRegistry,
} from 'ag-grid-community';
import {
  Search,
  Upload,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Cpu,
  FileText,
  Hash,
  Sun,
  Moon,
} from 'lucide-react';
import './index.css';
import './App.css';

ModuleRegistry.registerModules([AllCommunityModule]);

const API_BASE = '/api';

const commonThemeParams = {
  fontFamily: 'Inter, -apple-system, sans-serif',
  fontSize: 13,
};

const myThemeDark = themeQuartz.withPart(colorSchemeDarkBlue).withParams({
  backgroundColor: '#1c2128',
  foregroundColor: '#e6edf3',
  headerBackgroundColor: '#161b22',
  headerTextColor: '#8b949e',
  oddRowBackgroundColor: '#1c2128',
  rowHoverColor: '#21262d',
  borderColor: '#30363d',
  ...commonThemeParams
});

const myThemeLight = themeQuartz.withParams({
  backgroundColor: '#ffffff',
  foregroundColor: '#1F2328',
  headerBackgroundColor: '#f6f8fa',
  headerTextColor: '#656d76',
  oddRowBackgroundColor: '#ffffff',
  rowHoverColor: '#f3f4f6',
  borderColor: '#d0d7de',
  ...commonThemeParams
});

type ResultRow = {
  item_description: string;
  quantity?: string;
  country_of_origin?: string;
  hs_code: string;
  arabic_description: string;
  english_description: string;
  reasoning: string;
};

type SearchResult = {
  hs_code: string;
  desc_en: string;
  desc_ar: string;
  distance: number;
  hierarchy_path: string;
  duty_rate: string;
  procedures: string;
};

type Tab = 'classify' | 'search';



function DistanceBar({ value }: { value: number }) {
  // Gemini text embeddings typically return distances between ~0.4 (very high similarity) and ~0.85 (low similarity).
  // We map distance 0.45 to 100% and 0.85 to 0% for a more intuitive user experience.
  const mapped = (0.85 - value) / 0.40;
  const similarity = Math.max(0, Math.min(100, Math.round(mapped * 100)));

  const variant = similarity >= 75 ? 'success' : similarity >= 45 ? 'warning' : 'danger';
  return (
    <div className="distance-bar-wrap">
      <div className={`distance-bar-fill distance-bar-${variant}`} style={{ width: `${similarity}%` }} />
      <span className="distance-bar-label">{similarity}%</span>
    </div>
  );
}

function HierarchyBreadcrumb({ path }: { path: string }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  if (!path) return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;
  const segments = path.split(' > ');

  const handleMouseMove = (e: React.MouseEvent) => {
    if (tooltipRef.current) {
      tooltipRef.current.style.top = `${e.clientY - 12}px`;
      tooltipRef.current.style.left = `${e.clientX}px`;
    }
  };

  return (
    <>
      <div
        className="hierarchy-breadcrumb"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {segments.map((seg, i) => (
          <span key={i} style={{ display: 'contents' }}>
            {i > 0 && <span className="hierarchy-chevron">›</span>}
            <span className="hierarchy-segment">{seg}</span>
          </span>
        ))}
      </div>
      {showTooltip && createPortal(
        <div
          ref={tooltipRef}
          className="desc-tooltip"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            transform: 'translate(-50%, -100%)',
            zIndex: 10000,
            pointerEvents: 'none'
          }}
        >
          <div className="desc-tooltip-content">{path}</div>
          <div className="desc-tooltip-arrow" />
        </div>,
        document.body
      )}
    </>
  );
}

function DutyRateCell({ value }: { value: string }) {
  if (!value || value === 'nan') return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;
  const lower = value.toLowerCase();
  let variant = 'rate-default';
  if (lower.includes('exempt')) variant = 'rate-exempt';
  else if (lower.includes('%')) variant = 'rate-percent';
  else if (lower.includes('prohibit')) variant = 'rate-prohibited';
  return <span className={`duty-rate-cell ${variant}`}>{value}</span>;
}
// Official procedure descriptions from ZATCA API (eservices.zatca.gov.sa)
// Each entry has: ar = official Arabic text, en = concise English summary
const PROCEDURE_DATA: Record<string, { ar: string; en: string }> = {
  '1': { ar: 'يتطلب استيرادها وتصديرها موافقة نادي سباقات الخيل', en: 'Horse Racing Club approval required' },
  '2': { ar: 'يتطلب الموافقة المسبقة قبل الاستيراد من وزارة البيئة والمياه والزراعة وتقوم هيئة الزكاة والضريبة والجمارك بعرضها على الحجر النباتي والحيواني لاجازة فسحها من عدمه', en: 'Pre-approval from Ministry of Environment + Plant/Animal Quarantine' },
  '3': { ar: 'تعرض على المختبرات الخاصة', en: 'Subject to private laboratory testing' },
  '4': { ar: 'تعرض عند الاستيراد على الحجر النباتي والحيواني', en: 'Plant & Animal Quarantine inspection on import' },
  '7': { ar: 'تعرض عند الاستيراد على الحجر الزراعي', en: 'Agricultural Quarantine inspection on import' },
  '8': { ar: 'تعرض عند الاستيراد على الحجر الزراعي', en: 'Agricultural Quarantine inspection on import' },
  '9': { ar: 'يتطلب استيرادها وتصديرها موافقة الهيئة العامة للامن الغذائي', en: 'Food Security Authority approval required' },
  '10': { ar: 'تعرض على الهيئة العامة للغذاء والدواء', en: 'SFDA review required' },
  '12': { ar: 'تعرض على مختبر الهيئة السعودية للمواصفات والمقاييس والجودة', en: 'SASO laboratory testing required' },
  '13': { ar: 'يتطلب لاستيرادها وتصديرها موافقة وزارة الداخلية', en: 'Ministry of Interior approval required' },
  '15': { ar: 'تعرض عند الاستيراد على وزارة الاعلام', en: 'Ministry of Media review on import' },
  '16': { ar: 'إذا وردت للأغراض الزراعية تعرض على الحجر النباتي والحيواني (وزارة الزراعة)، وإذا كانت للأغراض المنزلية تعرض على الهيئة العامة للغذاء والدواء', en: 'Quarantine (agricultural) or SFDA (domestic use)' },
  '17': { ar: 'تستورد بموجب خطاب من قبل الجهات الحكومية المختصة', en: 'Requires letter from competent government authority' },
  '18': { ar: 'تعرض على البنك المركزي السعودي', en: 'Saudi Central Bank (SAMA) review' },
  '20': { ar: 'الأصناف العسكرية منها تفسح بموجب خطاب رسمي من الجهة العسكرية المستفيدة وفي حال تم استيرادها للشركات والمؤسسات فيتطلب لفسحها موافقة الهيئة العليا للأمن الصناعي بوزارة الداخلية', en: 'Military items: military authority letter; Companies: Industrial Security approval' },
  '21': { ar: 'يتطلب عند الاستيراد تقديم شهادة مطابقة من نظام سابر (الهيئة السعودية للمواصفات والمقاييس والجودة)', en: 'SABER conformity certificate required (SASO)' },
  '22': { ar: 'للأغراض الطبية والعلمية يتطلب موافقة الهيئة العامة للغذاء والدواء، أما للأغراض الخاصة بشركات ومؤسسات الصيانة والخدمات الفنية والمقاولات العامة يتطلب موافقة وزارة التجارة', en: 'SFDA (medical/scientific) or Ministry of Commerce (maintenance/services)' },
  '23': { ar: 'ممنوع استيرادها لغير القطاعات العسكرية (يتطلب لاستيرادها موافقة الجهة العسكرية المستفيدة)', en: 'Restricted to military sectors only' },
  '24': { ar: 'يتطلب فسح المواد الاولية (الكيميائية) من الجهات الحكومية ذات الاختصاص وذلك حسب نشاط الجهة المستوردة', en: 'Chemical raw materials: approval per importer activity type' },
  '26': { ar: 'ممنوع استيرادها لغير القطاعات العسكرية، وإذا وردت للجهات الحكومية يتطلب موافقة هذه الجهة', en: 'Restricted to military/government sectors' },
  '28': { ar: 'يتطلب تصدير المواشي (الأغنام، الماعز، الخيل، الإبل، البقر) موافقة من وزارة البيئة والمياه والزراعة', en: 'Livestock export: Ministry of Environment approval' },
  '29': { ar: 'يتطلب استيرادها وتصديرها موافقة مسبقة من المركز الوطني لتنمية الحياة الفطرية', en: 'National Wildlife Center pre-approval (CITES)' },
  '31': { ar: 'ممنوع تصدير أعلاف الماشية وهي الشعير والذرة البيضاء والسودانية والأعلاف الخضراء والتبن وكذلك اعلاف الدواجن وهي الذرة الصفراء وفول الصويا', en: 'Animal feed export prohibited' },
  '33': { ar: 'المشتقات البترولية يتطلب استيرادها أو تصديرها موافقة وزارة الطاقة', en: 'Ministry of Energy approval (petroleum derivatives)' },
  '35': { ar: 'يتطلب تصديرها موافقة وزارة الصناعة والثروة المعدنية', en: 'Ministry of Industry & Mineral Resources export approval' },
  '38': { ar: 'يتطلب عند الاستيراد تقديم شهادة مطابقة من نظام سابر', en: 'SABER conformity certificate required' },
  '39': { ar: 'يتطلب عند الاستيراد تقديم شهادة مطابقة من نظام سابر', en: 'SABER conformity certificate required' },
  '41': { ar: 'يتطلب لاستيرادها موافقة الهيئة العليا للأمن الصناعي', en: 'Industrial Security Authority approval required' },
  '42': { ar: 'يمنع تصدير مياه زمزم بموجب الأمر السامي الكريم إلا للحاج والمعتمر بواقع جالون واحد بسعة 5 لتر', en: 'Zamzam water export prohibited (except Hajj/Umrah pilgrims, 5L limit)' },
  '45': { ar: 'يتطلب لفسحها موافقة الهيئة العليا للأمن الصناعي بوزارة الداخلية، وفي حال تم استيرادها من قبل الجهات الحكومية يتم فسحها بموجب خطاب من الجهة المستفيدة', en: 'Industrial Security Authority (MOI) approval required' },
  '47': { ar: 'يتطلب لاستيرادها موافقة هيئة الاتصالات وتقنية المعلومات', en: 'CITC approval required (Telecom equipment)' },
  '49': { ar: 'يتطلب لاستيرادها وتصديرها موافقة وزارة الداخلية - إدارة الأسلحة والمتفجرات', en: 'MOI Arms & Explosives Dept. approval required' },
  '50': { ar: 'لا تستورد او تصدر هذه المواد الا من قبل جهات حكومية بموجب موافقه مسبقه من وزارة التجارة', en: 'Government entities only – Ministry of Commerce pre-approval' },
  '52': { ar: 'يتطلب استيرادها خطاب فسح من وزارة الداخلية (الهيئة العليا للأمن الصناعي - وحدة التراخيص الامنية) مبني على موافقة مكافحة المخدرات', en: 'MOI Industrial Security + Anti-Narcotics approval' },
  '54': { ar: 'يتطلب استيرادها وتصديرها موافقة المركز الوطني للرقابة على الالتزام البيئي', en: 'National Environmental Compliance Center approval' },
  '56': { ar: 'الوسائد الهوائية المحتوية على أي مادة ضمن مواد التي تدخل في تركيب المتفجرات يتطلب استيرادها موافقة الامن الصناعي', en: 'Airbags with explosive materials: Industrial Security approval' },
  '57': { ar: 'يلزم الحصول على رخصة تصدير من وزارة التجارة', en: 'Ministry of Commerce export license required' },
  '58': { ar: 'يتطلب لاستيرادها موافقة هيئة الاتصالات وتقنية المعلومات', en: 'CITC approval required (Smart cards/Telecom)' },
  '59': { ar: 'يشترط لإستيرادها و تصديرها الحصول على موافقة مسبقه من وزارة التجارة للشركات والمؤسسات، وبالنسبة للمصانع من وزارة الصناعه والثروه المعدنية', en: 'Pre-approval: Commerce (companies) or Industry (factories)' },
  '61': { ar: 'تعرض عند الاستيراد على الهيئة العامة للغذاء والدواء للموافقة على الفسح من عدمه', en: 'SFDA import clearance review' },
  '63': { ar: 'يشترط الحصول على موافقة وزارة الداخلية - إدارة الاسلحة والمتفجرات - عند الاستيراد والتصدير', en: 'MOI Arms & Explosives approval for import/export' },
  '64': { ar: 'يشترط لفسحها عند الاستيراد الحصول على موافقة الهيئة العامة للطيران المدني', en: 'GACA Civil Aviation Authority approval' },
  '65': { ar: 'تعرض على الجهات المختصة عند الاستيراد', en: 'Subject to competent authority review on import' },
  '66': { ar: 'تعرض عند الاستيراد على الهيئة العامة للغذاء والدواء', en: 'SFDA review on import (Water & Beverages)' },
  '67': { ar: 'يتطلب استيرادها أو تصديرها موافقة وزارة الطاقة', en: 'Ministry of Energy approval (Petroleum products)' },
  '68': { ar: 'يتطلب لاستيرادها موافقة الهيئة العليا للامن الصناعي "وحده التراخيص الامنية" للمواد التي تدخل في تركيب المتفجرات', en: 'Industrial Security – explosive precursor materials' },
  '71': { ar: 'تعرض عند الاستيراد على الهيئة العامة للغذاء والدواء', en: 'SFDA review on import (Poultry/Meat)' },
  '72': { ar: 'يتطلب تصديرها اجراء تحليل للتأكد من مطابقة المادة المصدرة مع التحليل', en: 'Export analysis required to verify conformity' },
  '74': { ar: 'فسح الاجهزة الضوئية والصوتية (سفتي واشارات انذار ضوئي) بخطاب من وزارة الداخلية - المرور', en: 'MOI Traffic Dept. clearance (vehicle light/sound equipment)' },
  '75': { ar: 'يتطلب تصديرها عرضها على ديوان هيئة الزكاة والضريبة والجمارك (الادارة العامة للتعرفة)', en: 'ZATCA Tariff Administration export review' },
  '76': { ar: 'تعرض عند الاستيراد على الهيئة العامة للغذاء والدواء', en: 'SFDA review on import (Tobacco)' },
  '77': { ar: 'يمنع دخول القرآن الكريم بأي شكل كان بما في ذلك المصاحف الإلكترونية والاقلام القارئة للقرآن الكريم (للكميات التجارية)', en: 'Commercial Quran import prohibited' },
  '78': { ar: 'يتطلب استيرادها وتصديرها موافقة المركز الوطني لتنمية الحياة الفطرية', en: 'National Wildlife Center approval (Birds of prey)' },
  '79': { ar: 'تعرض على الجهات المختصة', en: 'Subject to competent authority review' },
  '80': { ar: 'يتطلب عند الاستيراد تقديم شهادة كفاءة الطاقة', en: 'Energy Efficiency Registration (EER) certificate required' },
  '87': { ar: 'يلزم موافقة الهيئة العامة للغذاء والدواء (للسلائف الكيميائية عند الاستيراد)', en: 'SFDA approval for chemical precursors on import' },
  '91': { ar: 'يشترط لفسحها عند الاستيراد الحصول على موافقة الهيئة العامة للطيران المدني', en: 'GACA Civil Aviation Authority approval' },
  '92': { ar: 'يتطلب الحصول على رخصة تصدير من وزارة البيئة والمياه والزراعة', en: 'Ministry of Environment export license (Agriculture)' },
  '93': { ar: 'يشترط الحصول على إذن تصدير من وزارة البيئة والمياه والزراعة', en: 'Ministry of Environment export permit' },
  '94': { ar: 'يتطلب الحصول على تصريح من الإتحاد السعودي للهجن عند التصدير', en: 'Saudi Camel Racing Federation export permit' },
  '96': { ar: 'يتطلب عند الاستيراد تقديم شهادة مطابقة من نظام سابر', en: 'SABER conformity certificate required' },
  '97': { ar: 'يلزم موافقة المديرية العامة لمكافحة المخدرات (للسلائف الكيميائية عند التصدير)', en: 'Anti-Narcotics approval for chemical precursor export' },
  '98': { ar: 'يتطلب لتصديرها الحصول على موافقة من الهيئة العامة للغذاء والدواء', en: 'SFDA export approval required' },
  '99': { ar: 'يتطلب لتصديرها موافقة الهيئة العامة للأمن الغذائي', en: 'Food Security Authority export approval' },
  '100': { ar: 'يتطلب لتصديرها موافقة وزارة الطاقة', en: 'Ministry of Energy export approval' },
  '101': { ar: 'يلزم الحصول على موافقة وزارة الثقافة لاستيراد وتصدير القطع الأثرية', en: 'Ministry of Culture approval (Antiquities)' },
  '102': { ar: 'يشترط لاستيرادها وتصديرها الحصول على موافقة المركز الوطني لإدارة النفايات (موان)', en: 'National Waste Management Center (MAWAN) approval' },
  '104': { ar: 'يلزم الحصول على موافقة هيئة الرقابة النووية والإشعاعية عند الإستيراد أو التصدير', en: 'Nuclear & Radiological Regulatory Commission approval' },
  '105': { ar: 'يتطلب استيرادها موافقة وزارة الداخلية', en: 'Ministry of Interior import approval' },
  '106': { ar: 'يشترط لتصدير العينات للأغراض التشخيصية موافقة مكتب إرسال العينات إلى الخارج بوزارة الصحة، ويشترط لتصدير العينات للأغراض البحثية موافقة اللجنة الوطنية للأخلاقيات الحيوية بمدينة الملك عبدالعزيز للعلوم والتقنية', en: 'MOH (diagnostic samples) / KACST Bioethics (research samples) export approval' },
  '107': { ar: 'يتطلب عند الاستيراد تقديم شهادة مطابقة من نظام سابر', en: 'SABER conformity certificate required' },
  '108': { ar: 'يلزم الحصول على موافقة من وزارة المالية للتصدير', en: 'Ministry of Finance export approval' },
  '110': { ar: 'يشترط عند الاستيراد الحصول على موافقة المركز الوطني لإدارة النفايات (موان)', en: 'MAWAN approval for import (Scrap/Recycling)' },
  '111': { ar: 'يمنع استيرادها ويشترط لتصديرها الحصول على موافقة المركز الوطني لإدارة النفايات (موان)', en: 'Import prohibited; MAWAN export approval required' },
  '112': { ar: 'يتطلب لتصدير المواد البتروكيماوية موافقة وزارة الطاقة', en: 'Ministry of Energy approval (Petrochemical export)' },
  '113': { ar: 'لا تتطلب إذن مسبق لإتمام إجراءات فسوحات الاستيراد أو التصدير، إلا أن هذا الصنف يخضع لمتطلبات هيئة الرقابة النووية والإشعاعية في المجال النووي وللرقابة عليه', en: 'No pre-approval, but subject to Nuclear Regulatory Commission oversight' },
};

function ProcedureChip({ code }: { code: string }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const data = PROCEDURE_DATA[code];
  const enLabel = data?.en || `Customs Procedure ${code}`;
  const arText = data?.ar || '';

  const handleMouseMove = (e: React.MouseEvent) => {
    if (tooltipRef.current) {
      tooltipRef.current.style.top = `${e.clientY - 12}px`;
      tooltipRef.current.style.left = `${e.clientX}px`;
    }
  };

  return (
    <>
      <span
        className="procedure-chip procedure-chip-interactive"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {code}
      </span>
      {showTooltip && createPortal(
        <div
          ref={tooltipRef}
          className="procedure-tooltip"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            transform: 'translate(-50%, -100%)',
            zIndex: 10000,
            pointerEvents: 'none'
          }}
        >
          <div className="procedure-tooltip-content">
            <span className="procedure-tooltip-code">Proc. {code}</span>
            <span className="procedure-tooltip-desc">{enLabel}</span>
            {arText && <span className="procedure-tooltip-ar" dir="rtl">{arText}</span>}
          </div>
          <div className="procedure-tooltip-arrow" />
        </div>,
        document.body
      )}
    </>
  );
}

function DescriptionCell({ value }: { value: string }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const clean = value ? String(value).replace(/[-:]/g, '').replace(/\s+/g, ' ').trim().replace(/^(.)/, c => c.toUpperCase()) : '';
  if (!clean) return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;

  const handleMouseMove = (e: React.MouseEvent) => {
    if (tooltipRef.current) {
      tooltipRef.current.style.top = `${e.clientY - 12}px`;
      tooltipRef.current.style.left = `${e.clientX}px`;
    }
  };

  return (
    <>
      <span
        className="desc-cell-text"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {clean}
      </span>
      {showTooltip && createPortal(
        <div
          ref={tooltipRef}
          className="desc-tooltip"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            transform: 'translate(-50%, -100%)',
            zIndex: 10000,
            pointerEvents: 'none'
          }}
        >
          <div className="desc-tooltip-content">{clean}</div>
          <div className="desc-tooltip-arrow" />
        </div>,
        document.body
      )}
    </>
  );
}

function ProceduresCell({ value }: { value: string }) {
  if (!value || value === 'nan') return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;
  const codes = value.split(',').map(c => c.trim()).filter(Boolean);
  return (
    <div className="procedures-cell">
      {codes.map((code, i) => (
        <ProcedureChip key={i} code={code} />
      ))}
    </div>
  );
}

export default function App() {
  const [isLightMode, setIsLightMode] = useState(() => {
    if (typeof window !== 'undefined') {
      const savedTheme = sessionStorage.getItem('theme-preference');
      if (savedTheme) {
        return savedTheme === 'light';
      }
      if (window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: light)').matches;
      }
    }
    return false;
  });

  const [tab, setTab] = useState<Tab>('search');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; value?: string; action: 'copy' | 'reset-layout'; openedAt: number } | null>(null);
  const gridApiRef = useRef<any>(null);
  const searchAbortRef = useRef<AbortController | null>(null);
  const classifyAbortRef = useRef<AbortController | null>(null);

  // Classify tab state
  const [rawInput, setRawInput] = useState(
    'Kugellager für Industriemotor\nEdelstahl-Kreiselpumpe für Wasser\nKupferkabel 2.5mm'
  );
  const [results, setResults] = useState<ResultRow[]>([]);

  // Search tab state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  useEffect(() => {
    const handleClick = () => setContextMenu(null);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (isLightMode) {
      root.classList.add('theme-light');
      sessionStorage.setItem('theme-preference', 'light');
    } else {
      root.classList.remove('theme-light');
      sessionStorage.setItem('theme-preference', 'dark');
    }
  }, [isLightMode]);

  const clearMessages = () => { setError(''); setSuccess(''); };

  const handleClassify = async () => {
    clearMessages();
    const lines = rawInput.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) { setError('Please enter at least one item description.'); return; }

    // Cancel any previous in-flight classify request
    classifyAbortRef.current?.abort();
    const controller = new AbortController();
    classifyAbortRef.current = controller;

    const items = lines.map(l => ({ item_description: l }));
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      setResults(data.results);
      setSuccess(`✓ ${data.results.length} items classified successfully.`);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') return; // Request was superseded
      setError(e instanceof Error ? e.message : 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    clearMessages();
    if (!searchQuery.trim()) { setError('Please enter a search query.'); return; }

    // Cancel any previous in-flight search request
    searchAbortRef.current?.abort();
    const controller = new AbortController();
    searchAbortRef.current = controller;

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, top_k: 50 }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      const filteredData = data.filter((item: SearchResult) => {
        const mapped = (0.85 - item.distance) / 0.40;
        const similarity = Math.max(0, Math.min(100, Math.round(mapped * 100)));
        return similarity > 20;
      });
      setSearchResults(filteredData);
      setSuccess(`Found ${filteredData.length} matches.`);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') return; // Request was superseded
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    const res = await fetch(`${API_BASE}/export`);
    if (!res.ok) { setError('No export available. Run classification first.'); return; }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hs_classified.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const cleanDesc = (val: any) => {
    if (!val) return '';
    return String(val).replace(/[-:]/g, '').replace(/\s+/g, ' ').trim().replace(/^(.)/, c => c.toUpperCase());
  };

  const classifyColDefs: any[] = [
    { field: 'item_description', headerName: 'Item Description', flex: 2, minWidth: 200 },
    {
      field: 'hs_code', headerName: 'HS Code', width: 170,
      cellRenderer: ({ value }: { value: string }) => (
        <span className="hs-code-cell">{value}</span>
      )
    },
    {
      field: 'english_description', headerName: 'English Description (EN)', flex: 2, minWidth: 200,
      cellRenderer: ({ value }: { value: string }) => <DescriptionCell value={value} />,
    },
    {
      field: 'arabic_description', headerName: 'Arabic Description (AR)', flex: 2, minWidth: 200,
      valueFormatter: (p: any) => cleanDesc(p.value)
    },
  ];

  const searchColDefs: any[] = [
    {
      field: 'hs_code', headerName: 'HS Code', width: 160,
      cellRenderer: ({ value }: { value: string }) => (
        <span className="hs-code-cell">{value}</span>
      )
    },
    {
      field: 'desc_en', headerName: 'English Description', flex: 1.5, minWidth: 150,
      cellRenderer: ({ value }: { value: string }) => <DescriptionCell value={value} />,
    },
    {
      field: 'desc_ar', headerName: 'Arabic Description', flex: 1.5, minWidth: 150,
      valueFormatter: (p: any) => cleanDesc(p.value)
    },
    {
      field: 'hierarchy_path', headerName: 'Category Path', flex: 2, minWidth: 220,
      autoHeight: true,
      cellStyle: { lineHeight: '1.4', padding: '4px 8px' },
      cellRenderer: ({ value }: { value: string }) => <HierarchyBreadcrumb path={value} />,
    },
    {
      field: 'duty_rate', headerName: 'Duty', width: 130,
      cellRenderer: ({ value }: { value: string }) => <DutyRateCell value={value} />,
    },
    {
      field: 'procedures', headerName: 'Procedures', width: 140,
      cellRenderer: ({ value }: { value: string }) => <ProceduresCell value={value} />,
    },
    {
      field: 'distance', headerName: 'Similarity Score', width: 160,
      resizable: false,
      cellRenderer: ({ value }: { value: number }) => <DistanceBar value={value} />,
    },
  ];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <a href="/" className="header-brand" aria-label="SaHS Home">
            <img src="/SaHS_logo.png" alt="SaHS Logo" className="brand-logo" />
            <span className="brand-sub">Saudi HS Code Intelligence</span>
          </a>
          <div className="header-actions">
            <button
              className="btn btn-icon"
              onClick={() => setIsLightMode(!isLightMode)}
              title={isLightMode ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
            >
              {isLightMode ? <Moon size={18} /> : <Sun size={18} />}
            </button>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="tabs">
        <div className="tabs-inner">
          <button className={`tab-btn ${tab === 'search' ? 'active' : ''}`} onClick={() => { setTab('search'); clearMessages(); }}>
            <Search size={15} /> Search HS Codes
          </button>
          <button className={`tab-btn ${tab === 'classify' ? 'active' : ''}`} onClick={() => { setTab('classify'); clearMessages(); }}>
            <FileText size={15} /> Classify Invoice Items
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="main">

        {/* Toast messages */}
        {error && (
          <div className="toast toast-error">
            <AlertCircle size={15} /> {error}
          </div>
        )}
        {success && (
          <div className="toast toast-success">
            <CheckCircle2 size={15} /> {success}
          </div>
        )}

        {/* CLASSIFY TAB */}
        {tab === 'classify' && (
          <>
            <div className="card">
              <div className="card-header">
                <h2 className="card-title"><FileText size={16} /> Invoice Items</h2>
                <span className="card-hint">Enter one item per line — any language (German, English, Arabic)</span>
              </div>
              <textarea
                className="textarea"
                rows={5}
                value={rawInput}
                onChange={e => setRawInput(e.target.value)}
                placeholder="Kugellager für Industriemotor&#10;Edelstahl-Kreiselpumpe für Wasser&#10;Kupferkabel 2.5mm"
              />
              <div className="card-actions">
                <button className="btn btn-primary" onClick={handleClassify} disabled={loading}>
                  {loading ? <Loader2 size={15} className="spin" /> : <Cpu size={15} />}
                  {loading ? 'Classifying...' : 'Classify'}
                </button>
                {results.length > 0 && (
                  <button className="btn btn-secondary" onClick={handleExport}>
                    <Download size={15} /> Export CSV
                  </button>
                )}
              </div>
            </div>

            {results.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title"><Hash size={16} /> Classification Results</h2>
                  <span className="card-hint">{results.length} items classified</span>
                </div>
                <div className="grid-wrap" onContextMenu={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest('.ag-header-cell-resize') || target.classList.contains('ag-header-cell-resize')) {
                    e.preventDefault();
                    setContextMenu({ x: e.clientX, y: e.clientY, action: 'reset-layout', openedAt: Date.now() });
                  }
                }}>
                  <AgGridReact
                    theme={isLightMode ? myThemeLight : myThemeDark}
                    rowData={results}
                    columnDefs={classifyColDefs}
                    domLayout="autoHeight"
                    defaultColDef={{ sortable: true, filter: true, resizable: true }}
                    preventDefaultOnContextMenu={true}
                    onGridReady={(params) => { gridApiRef.current = params.api; }}
                    onCellContextMenu={(e) => {
                      if (e.event) {
                        e.event.preventDefault();
                        setContextMenu({ x: (e.event as MouseEvent).clientX, y: (e.event as MouseEvent).clientY, value: e.value ? String(e.value) : '', action: 'copy', openedAt: Date.now() });
                      }
                    }}
                  />
                </div>
              </div>
            )}

            {/* File Upload section moved here */}
            <div className="card upload-card">
              <div className="upload-icon"><Upload size={24} /></div>
              <div>
                <h3 className="upload-title">Bulk CSV Upload</h3>
                <p className="upload-desc">Upload a CSV with an <code>item_description</code> column to classify thousands of invoice lines at once.</p>
              </div>
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  clearMessages();
                  setLoading(true);
                  const formData = new FormData();
                  formData.append('file', f);
                  try {
                    const res = await fetch(`${API_BASE}/classify/csv`, { method: 'POST', body: formData });
                    if (!res.ok) throw new Error((await res.json()).detail);
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'classified_output.csv';
                    a.click();
                    URL.revokeObjectURL(url);
                    setSuccess('Bulk classification complete — your download started.');
                  } catch (e: unknown) {
                    setError(e instanceof Error ? e.message : 'Upload failed');
                  } finally {
                    setLoading(false);
                  }
                }}
              />
              <label htmlFor="csv-upload" className="btn btn-secondary" style={{ cursor: 'pointer' }}>
                <Upload size={15} /> Upload CSV
              </label>
            </div>
          </>
        )}

        {/* SEARCH TAB */}
        {tab === 'search' && (
          <>
            <div className="card">
              <div className="card-header">
                <h2 className="card-title"><Search size={16} /> Vector Search</h2>
                <span className="card-hint">Search using any language — the model handles English, Arabic and German</span>
              </div>
              <div className="search-row">
                <input
                  className="input"
                  type="text"
                  placeholder="e.g. ball bearing for industrial motor"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSearch()}
                />
                <button className="btn btn-primary" onClick={handleSearch} disabled={loading}>
                  {loading ? <Loader2 size={15} className="spin" /> : <Search size={15} />}
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>

            {searchResults.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title"><Hash size={16} /> Search Results</h2>
                </div>
                <div className="grid-wrap" onContextMenu={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest('.ag-header-cell-resize') || target.classList.contains('ag-header-cell-resize')) {
                    e.preventDefault();
                    setContextMenu({ x: e.clientX, y: e.clientY, action: 'reset-layout', openedAt: Date.now() });
                  }
                }}>
                  <AgGridReact
                    theme={isLightMode ? myThemeLight : myThemeDark}
                    rowData={searchResults}
                    columnDefs={searchColDefs}
                    domLayout="autoHeight"
                    defaultColDef={{ sortable: true, filter: true, resizable: true }}
                    preventDefaultOnContextMenu={true}
                    onGridReady={(params) => { gridApiRef.current = params.api; }}
                    onCellContextMenu={(e) => {
                      if (e.event) {
                        e.event.preventDefault();
                        setContextMenu({ x: (e.event as MouseEvent).clientX, y: (e.event as MouseEvent).clientY, value: e.value ? String(e.value) : '', action: 'copy', openedAt: Date.now() });
                      }
                    }}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="footer">
        SaHS · Powered by Gemini + ChromaDB · <button onClick={() => setShowDisclaimer(true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', textDecoration: 'underline', padding: 0, font: 'inherit' }}>Disclaimer</button>
        <a
          href="https://github.com/donrami/SaHS"
          target="_blank"
          rel="noopener noreferrer"
          className="footer-github-btn"
          title="View on GitHub"
          aria-label="View SaHS on GitHub"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            width="18"
            height="18"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
          </svg>
          GitHub
        </a>
      </footer>

      {/* Disclaimer Overlay */}
      {showDisclaimer && (
        <div className="modal-overlay" onClick={() => setShowDisclaimer(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="card-header" style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 className="card-title"><AlertCircle size={16} /> Disclaimer</h2>
              <button className="btn btn-icon" onClick={() => setShowDisclaimer(false)} aria-label="Close modal">
                <span style={{ fontSize: '20px', lineHeight: 1 }}>&times;</span>
              </button>
            </div>
            <p style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
              The HS code classifications and related information provided by this tool are generated by artificial intelligence and are intended for informational purposes only. While we strive for accuracy, these classifications may not be complete or legally binding. Formal bindings and definitive HS code classifications must be obtained directly from the relevant customs authorities. We disclaim any liability for customs clearance issues, duties, or penalties resulting from the use of this service.
            </p>
            <div style={{ marginTop: '24px', textAlign: 'right' }}>
              <button className="btn btn-primary" onClick={() => setShowDisclaimer(false)}>I Understand</button>
            </div>
          </div>
        </div>
      )}

      {/* Context Menu for Grid Cells */}
      {contextMenu && (
        <div
          style={{
            position: 'fixed',
            top: contextMenu.y,
            left: contextMenu.x,
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '4px',
            zIndex: 9999,
            boxShadow: 'var(--shadow)',
            display: 'flex',
            flexDirection: 'column'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            style={{
              padding: '8px 16px',
              background: 'transparent',
              border: 'none',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              textAlign: 'left',
              fontSize: '13px',
              fontFamily: 'inherit',
              borderRadius: '4px'
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-tertiary)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            onPointerUp={(e) => {
              if (e.button !== 0) return; // Explicitly only allow left-clicks

              if (contextMenu.action === 'reset-layout') {
                if (gridApiRef.current) {
                  gridApiRef.current.resetColumnState();
                }
                // No status update message for column layout reset
              } else if (contextMenu.action === 'copy' && contextMenu.value) {
                navigator.clipboard.writeText(contextMenu.value);
                setSuccess('Copied to clipboard');
              }
              setContextMenu(null);
            }}
          >
            {contextMenu.action === 'reset-layout' ? 'Reset Column Layout' : 'Copy cell contents'}
          </button>
        </div>
      )}
    </div>
  );
}
