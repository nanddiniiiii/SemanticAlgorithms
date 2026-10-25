"""Build an evidence-backed Stage1 report from persisted receipts, never paper scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/'Docs'
OUT = ROOT/'output/pdf'


def read(name):
    path = DOCS/'validation'/name
    return json.loads(path.read_text()) if path.exists() else None


def main():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.units import inch
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat(timespec='seconds')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    inventory = read('stage1_results_inventory.json')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=29, leading=34, textColor=colors.HexColor('#142b43'), spaceAfter=18))
    styles.add(ParagraphStyle(name='Deck', fontSize=12, leading=18, textColor=colors.HexColor('#42586c'), spaceAfter=14))
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 9
    styles['Heading1'].textColor = colors.HexColor('#142b43')
    styles['Heading2'].textColor = colors.HexColor('#147d92')
    styles.add(ParagraphStyle(name='Cell', fontSize=8, leading=10, alignment=TA_LEFT))
    story, md = [], ['# Stage 1 experimental report', '', 'Snapshot: '+snapshot, 'Source checkpoint: `'+revision+'`', '']

    def heading(title):
        story.append(Paragraph(escape(title), styles['Heading1'])); md.extend(['## '+title, ''])

    def paragraph(text):
        story.append(Paragraph(escape(text), styles['BodyText'])); md.extend([text, ''])

    def table(headers, rows, widths):
        cells = [[Paragraph(escape(str(v)), styles['Cell']) for v in row] for row in [headers]+rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceef2')),
            ('VALIGN',(0,0),(-1,-1),'TOP'), ('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7), ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#147d92')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')])]))
        story.extend([item, Spacer(1,12)])
        md.extend(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        md.extend('| '+' | '.join(str(v).replace('\n','; ') for v in row)+' |' for row in rows); md.append('')

    story.append(Paragraph('Semantic image communication', styles['ReportTitle']))
    story.append(Paragraph('Stage 1 | Existing-method baselines and execution evidence', styles['Deck']))
    paragraph('This report consolidates retained models, author checkpoints, executed evaluations and missing-weight training. It is a measured progress report, not a claim that every requested paper has been reproduced or that Stage 1 is complete.')
    paragraph('Prepared '+snapshot+'. Source checkpoint '+revision+'. Generated receipts and the report may include uncommitted work after this checkpoint; each remote source bundle has its own hash in KAGGLE_RUNS.md.')
    heading('Decisions and evidence')
    paragraph('Retain existing checkpoints unchanged. Prefer verified author weights to unnecessary retraining. Train absent weights only when the paper variant, implementation, dimensions, data split and channel are defensible. Stage 2 dataset adaptation and diffusion training remain outside this execution.')
    paragraph('Two Kaggle accounts were used concurrently. Completed and recovered: the 200-epoch STL10 classifier prerequisite, two native SwinJSCC Kodak evaluations, ADJSCC C2 quality/profiles, JCM GPU profiles, NTSCC GPU quality/profiles and SQ-GAN metric extension. Shared codec construction/training and custom-data adaptation are separate next runs; current versions and status are recorded in KAGGLE_RUNS.md. A classifier alone does not establish a trained semantic codec.')
    paragraph('Complete metric coverage is still pending. Missing values mean unmeasured or unsupported, never zero. No model in this report establishes guaranteed lossless one-shot delivery.')
    heading('Reproducible comparison contract')
    table(['Axis','Recorded contract'], [
        ['Image','RGB layout, tensor dimensions, native preprocessing, dataset and split; no training on Kodak'],
        ['Channel','AWGN or native digital chain, SNR definition, noise seed/trials and normalization; ideal source codecs kept separate'],
        ['Rate','Actual complex uses, CBR = k/(3HW), or payload bits and bpp; headers, side streams, pilots and protection distinguished'],
        ['Quality','PSNR, SSIM, MS-SSIM, LPIPS and exact-pixel checks where available; semantic accuracy only with task evidence'],
        ['Compute','Inference only; encoder, decoder and full scope; parameters, counted operators and omissions'],
        ['Runtime','Actual device, precision, batch, dimensions, warmup/repeats, synchronized latency and memory scope']
    ], [65,439])
    story.append(PageBreak())
    heading('Measured quality results')
    rows = []
    for label, name, psnr, ssim, data in [
        ('Swin fixed C32','swin_fixed_c32_gpu_metrics.json','mean_psnr_db','mean_ssim','Kodak24; 10 dB AWGN'),
        ('Swin SA+RA C32','swin_sa_ra_gpu_metrics.json','mean_psnr_db','mean_ssim','Kodak24; 10 dB AWGN'),
        ('ADJSCC C2','adjscc_gpu_metrics.json','mean_psnr_db','mean_ssim','Kodak24; 10 dB AWGN; declared lineage'),
        ('NTSCC q1','ntscc_gpu_metrics.json','mean_psnr_db','mean_ssim','Kodak24; 10 dB AWGN; incomplete side rate'),
        ('JCM 16-QAM','jcm_pretrained_metrics.json','standard_rgb_psnr_db','author_ssim','CIFAR10 test10k; 12 dB'),
        ('SQ-GAN','sqgan_cityscapes500_native_metrics.json','mean_psnr_native_saved_range1_db','mean_ssim','Cityscapes val500; ideal codec')]:
        receipt = read(name)
        if receipt:
            rows.append([label, data, f'{receipt[psnr]:.4f}', f'{receipt[ssim]:.5f}' if ssim else 'Unmeasured',
                f"{receipt['mean_ms_ssim']:.5f}" if receipt.get('mean_ms_ssim') is not None else 'Unmeasured',
                f"{receipt['mean_lpips']:.5f}" if receipt.get('mean_lpips') is not None else 'Unmeasured'])
    table(['Model','Native workload','PSNR dB','SSIM','MS-SSIM','LPIPS'], rows, [86,166,63,63,63,63])
    paragraph('These are separate native evaluations, not a matched leaderboard. JCM SSIM preserves the author convention, including its unusual range setting; it cannot be pooled with standard RGB SSIM. SQ-GAN PSNR uses native saved-image postprocessing. Its conventional rounded-image PSNR is 21.2566 dB.')
    paragraph('Measured semantic fidelity: JCM classification accuracy is 87.64% on the official 10,000-image CIFAR10 test split. SQ-GAN decoded semantic-branch mIoU is 0.740114 against supplied paired labels; this is not segmentation of reconstructed RGB or the paper InternImage metric. Kodak has no independent task annotations, so semantic metrics remain unmeasured.')
    paragraph('Swin fixed-C32 reaches 29.3711 dB, while the adaptive SA+RA checkpoint reaches only 21.3327 dB at the same requested C32 and 10 dB. This large gap is an unresolved checkpoint/operating-point result, not proof that adaptation is generally worse. Strict state loading is necessary but does not certify expected published performance. Keep the adaptive result flagged until checkpoint configuration and rate/SNR sweeps explain it.')
    heading('Rate accounting and exactness')
    table(['Model','Main-stream budget','Limit'],[
        ['Swin C32 / ADJSCC C2','k = 24,576 complex uses at 512 x 768; CBR 1/48 = 0.0208333','Transport, pilot and side overhead unspecified'],
        ['NTSCC q1','GPU mean CBR 0.02397537; content-dependent main stream','Rate-index/mask transport unmodeled; total rate uncertified'],
        ['JCM','128 complex uses / 32 x 32 RGB image; CBR 1/24','Native 16-QAM; 12 dB; cannot compare directly with Kodak'],
        ['SQ-GAN','3,584 paper payload bits; 0.02734375 bpp; SQ01 packet 3,712 bits','No noisy channel/FEC; bit budget is not complex-use budget']
    ], [78,235,191])
    paragraph('CBR is complex channel uses per RGB scalar sample in these JSCC records. It is not file compression ratio, bpp, bandwidth in hertz or a packet count. Converting digital bits to radio resources needs modulation order, code rate and overhead. Semantic similarity and perceptual quality do not establish exact pixels or exact files.')
    paragraph('NTSCC correction: decoder hyperprior refinement is disabled in this retained configuration. The decoder is given spatial rate indices and a scatter mask without modeling their transport. With 16 choices and 32 x 48 Kodak latent positions, a fixed-width index representation costs 6,144 uncoded bits (768 bytes, 0.015625 bpp). This is not a minimum rate or a radio-use count. Encoder-derived receive rescaling also assumes an unmodeled available power value.')
    traditional=read('traditional_kodak24_metrics.json')
    if traditional:
        story.append(PageBreak())
        heading('Traditional source-codec reference')
        paragraph('All 24 Kodak images are decoded as native RGB. These measurements include complete encoded file bytes (except headerless RAW), with no noisy channel, modulation or FEC. They establish a source-compression reference; they cannot be ranked against neural radio CBR without a common digital transmission chain.')
        table(['Codec setting','Mean bpp','RGB PSNR dB','SSIM','Exact images'],
            [[row['variant'],f"{row['mean_bpp']:.4f}",
                'Infinite (exact)' if row['psnr_is_infinite'] else f"{row['mean_psnr_db']:.4f}",
                f"{row['mean_ssim']:.5f}",str(row['exact_rgb_images'])+'/24'] for row in traditional['variants']],
            [172,76,104,76,76])
        paragraph('PNG and reversible JPEG2000 recover every RGB sample in this test. They do not reproduce the original PNG container bytes. Lossy JPEG uses Pillow quality settings with 4:2:0 chroma subsampling; lossy JPEG2000 uses OpenJPEG rate targets and an irreversible transform. Actual encoded bytes, not requested targets, determine bpp.')
        paragraph('On this Kodak set, the lossless file rates are about 13.4 bpp versus 24 bpp RAW, approximately 1.8 times smaller. This is an empirical result for these images and settings, not a universal compression limit. CPU encode/decode observations are saved, but one timing observation per image is not a stable latency distribution or a GPU comparison.')
    legacy=read('deepjscc_legacy_kodak24_metrics.json')
    if legacy:
        heading('Retained DeepJSCC diagnostic')
        paragraph(f"The historical retained checkpoint gives {legacy['mean_psnr_db']:.4f} dB PSNR, SSIM {legacy['mean_ssim']:.5f} and MS-SSIM {legacy['mean_ms_ssim']:.5f} on Kodak24 using 128 x 128 tiles and its legacy real AWGN channel at 10 dB. It was trained at 200 dB. Each tile produces 19 x 29 x 29 = 15,979 real values; a full Kodak image produces 383,496. No explicit complex packing/radio protocol is implemented. This is a diagnostic of this checkpoint and port, not an estimate of the original DeepJSCC paper's performance. It remains outside the matched-radio comparison.")
    story.append(PageBreak())
    heading('Measured GPU inference profiles')
    profile_rows = []
    for prefix, label in [('swin_fixed_c32','Swin fixed C32'),('swin_sa_ra','Swin SA+RA'),('adjscc','ADJSCC C2'),('ntscc','NTSCC q1'),('jcm','JCM32x32'),('sqgan','SQ-GAN')]:
        for component in ['encoder','decoder','full']:
            p = read(prefix+'_profile_'+component+'.json')
            if not p: continue
            profile_rows.append([label+' / '+component, f"{p['parameters']/1e6:.3f}",
                f"{p['forward_flops']/1e9:.2f}" if p.get('forward_flops') is not None else 'Unavailable',
                f"{p['latency_p50_seconds']*1000:.2f}",f"{p['latency_p95_seconds']*1000:.2f}" if p.get('latency_p95_seconds') else 'Unmeasured',
                f"{p['incremental_peak_cuda_allocated_bytes']/2**20:.1f}" if p.get('incremental_peak_cuda_allocated_bytes') is not None else 'Unmeasured'])
    table(['Component','M params','Partial GFLOPs','p50 ms','p95 ms','Extra MiB'],profile_rows,[141,63,80,65,65,90])
    paragraph('Swin profiles use one T4, FP32, batch 1, 512 x 768 RGB, three warmups and ten repeats. Two T4s were allocated; this native inference path uses CUDA0. Parameter counts are component-specific; whole-model memory remains resident for component profiles. Extra MiB is peak allocation above the warmed baseline, not deployment memory. LPIPS and MS-SSIM evaluation are outside timed forwards; Swin native full forward still includes its own distortion calculations.')
    paragraph('Additional profile contracts: ADJSCC and NTSCC use 512 x 768, three warmups and twenty repeats. NTSCC full scope includes auxiliary NTC reconstruction and JSCC; repeats reuse seeded channel noise. JCM rows use 32 x 32, ten warmups and one hundred repeats, including both reconstruction and classification; batch32 profiles are separate. SQ-GAN uses paired RGB/semantic 256 x 512, three warmups and twenty repeats, with both branches included. These workload differences prevent a uniform latency ranking.')
    paragraph('The profiler counts only supported operations and is not a total FLOPs certification. A separate registered-kernel counter adds coverage for transposed convolutions and emits observed uncounted operator types. Its reference tests check a 1,728-FLOP transposed convolution and a 48-FLOP linear layer. Never add the two counters together or compare incompatible scopes as speedups.')
    ad_count = read('adjscc_profile_full.json')
    if ad_count and ad_count.get('registered_kernel_count'):
        paragraph(f"ADJSCC illustrates the coverage problem: the standard profiler reports {ad_count['forward_flops']/1e9:.2f} GFLOPs, while the registered-kernel counter reports {ad_count['registered_kernel_count']['registered_kernel_flops']/1e9:.2f} GFLOPs for the same full forward. The latter includes transposed convolution; unregistered elementwise work remains omitted. Ranking this model against Swin using only the smaller profiler number would be misleading.")
    paragraph('Earlier CPU profiles exist for DeepJSCC, NTSCC, ADJSCC, JCM and SQ-GAN. They are computational diagnostics, not T4 latency comparisons. Training FLOPs have not been measured. Runtime reliability here means observed completion and finite outputs, not a certified radio BLER or a large repeated-noise trial.')
    heading('Training status and scientific boundaries')
    paragraph('Shared Codebook: native STL10 classifier prerequisite uses 5,000 labelled training images, 96 x 96 input, batch 64, FP32 and the source learning-rate schedule, capped at 200 epochs or 7,200 seconds. A two-step real-data CUDA pilot and strict save/reload gate precede training. The classifier must then feed audited codebook construction and codec training; these stages cannot be skipped.')
    classifier = read('shared_classifier_training.json')
    if classifier:
        paragraph(f"Completed classifier evidence: {classifier['epochs_entered']} epochs, {classifier['optimizer_steps']:,} optimizer steps, {classifier['training_images_seen']:,} training-image presentations in {classifier['elapsed_seconds']:.1f} seconds. Strict checkpoint reload passed. Final training accuracy is {classifier['last_epoch']['training_accuracy']*100:.2f}% and loss {classifier['last_epoch']['training_loss']:.6f}; these are training-set metrics, not generalization or semantic-codec results.")
    paragraph('DeepJSCC-Q remains blocked: the public candidate mixes paper variants and has verified quantizer evaluation and regularizer defects. No long training or faithful trained baseline is claimed.')
    story.append(PageBreak())
    heading('All requested methods: executable status')
    names = {'deepjscc':'DeepJSCC','swinjscc':'SwinJSCC','ntscc':'NTSCC','jcm':'JCM','adjscc':'ADJSCC',
        'shared_semantic_codebook':'Shared Semantic-Aware Codebook','sqgan':'SQ-GAN','vq_deepsc':'VQ-DeepSC',
        'vq_deepisc':'VQ-DeepISC','sc_lmkb':'SC-LMKB','mdjcm':'MDJCM','mambajscc':'MambaJSCC',
        'diffcom':'DiffCom','diffjscc':'DiffJSCC','deepjscc_q':'DeepJSCC-Q',
        'hybrid_semantic_complementary':'Our complementary hybrid','dit_jscc':'DiT-JSCC',
        'hda_deepsc':'HDA-DeepSC','masked_vqvae':'Masked VQ-VAE','moc_rvq':'MOC-RVQ'}
    table(['Algorithm','Stage 1 category','Implementation boundary'],
        [[names.get(row['algorithm'],row['algorithm']),row.get('action_group','Unknown').replace('_',' '),row.get('fidelity_note','')]
         for row in inventory['models']], [102,135,267])
    heading('Failure recovery and controls')
    paragraph('Earlier A/B jobs allocated T4 hardware but failed before model execution because their source archive omitted sources.lock.json. The corrected bundle is extracted and imported in a fresh process during preflight. Stage logs now retain start/end, elapsed time, traceback and source hash. Failed runs remain archived; they produce no quality or training claim.')
    paragraph('The retained Swin port had a masked-power normalization error that made nominal and effective SNR disagree. Its quality results remain quarantined. Native author checkpoints are evaluated separately. NTSCC side-rate incompleteness, ADJSCC implementation lineage and SQ-GAN label dependence remain explicit limitations.')
    paragraph('Retrieved files, not a running-cell display or /kaggle/working alone, establish saved evidence. Script-kernel logs, JSON/CSV files and reconstructions were recovered. Executed notebook/HTML exports were not exposed for the completed Swin run; this capture limitation is recorded rather than concealed.')
    heading('Completion gates and next decisions')
    paragraph('Stage 1 closes per algorithm only after checkpoint provenance, native input/output contract, finite inference, quality/rate records, inference profiles and representative artifact recovery are checked, or after an explicit unavailable/blocked disposition. It is not honest to mark missing native implementations as completed models.')
    paragraph('Remaining Phase1 work: resolve the adaptive Swin anomaly; progress Shared Codebook through construction, pilot and bounded native-branch training; validate Mamba CUDA compatibility; keep absent releases and DeepJSCC-Q fidelity blockers explicit. Custom-data adaptation is now authorized separately, with frozen train/development/test membership and untouched native checkpoints. Dataset or architecture changes must not be presented as native-paper reproduction.')
    heading('Evidence and reproduction map')
    paragraph('Canonical plan: Docs/STAGE_ONE_BASELINES.md. Remote run identities, versions and retrieval: Docs/KAGGLE_RUNS.md. Machine-readable inventory: Docs/validation/stage1_results_inventory.json. Per-run events/checkpoints: Results/experiments/<algorithm>/<run-id>/. Canonical appended metric records: Results/metrics.csv. Author checkpoint sources: Docs/PRETRAINED_DOWNLOADS.md. All paths are relative to this repository.')
    paragraph('Rebuild inventory: python scripts/build_stage1_results.py. Import a recovered run: python scripts/import_stage1_outputs.py --input <version-specific-folder>. Rebuild this report with the managed Python runtime: python scripts/build_stage1_report.py. Fresh remote runs require their own recorded source revision, bundle hash, inputs and actual hardware check.')
    (DOCS/'STAGE_ONE_REPORT.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#42586c'))
        canvas.drawString(45,28,'SemanticLossless | Stage 1 | Measured evidence and limitations')
        canvas.drawRightString(555,28,str(doc.page))
    target = OUT/'SemanticLossless_Stage1_Report.pdf'
    SimpleDocTemplate(str(target), pagesize=(600,842), leftMargin=48,rightMargin=48,
        topMargin=43,bottomMargin=48,title='SemanticLossless Stage 1 Experimental Report',author='SemanticLossless research project').build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)


if __name__ == '__main__':
    main()
