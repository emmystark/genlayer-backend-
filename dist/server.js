import dns from 'node:dns';
dns.setDefaultResultOrder('ipv4first');
import express from 'express';
import multer from 'multer';
import dotenv from 'dotenv';
import mongoose, { Schema, model } from 'mongoose';
import { v2 as cloudinary } from 'cloudinary';
import { Readable } from 'stream';
import Groq from 'groq-sdk';
import fs from 'fs/promises';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import ffprobeInstaller from '@ffprobe-installer/ffprobe';
// Set paths
ffmpeg.setFfmpegPath(ffmpegInstaller.path);
ffmpeg.setFfprobePath(ffprobeInstaller.path);
// Multer MUST come AFTER the above, but BEFORE routes
const upload = multer({
    storage: multer.diskStorage({
        destination: (req, file, cb) => cb(null, '/tmp'),
        filename: (req, file, cb) => {
            cb(null, `${Date.now()}-${file.originalname.replace(/\s+/g, '-')}`);
        },
    }),
    limits: { fileSize: 100 * 1024 * 1024 }, // 100MB
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/') || file.mimetype.startsWith('video/')) {
            cb(null, true);
        }
        else {
            cb(new Error('Only images and videos allowed'));
        }
    }
});
// Flexible upload (handles both 'image' and 'media')
// Replace your current flexibleUpload with this error-catching version
const flexibleUpload = (req, res, next) => {
    upload.fields([
        { name: 'media', maxCount: 1 },
        { name: 'image', maxCount: 1 },
        { name: 'file', maxCount: 1 },
    ])(req, res, (err) => {
        if (err) {
            console.error('❌ Multer error:', err.message, err.code);
            return res.status(400).json({
                error: 'File upload failed',
                detail: err.message,
            });
        }
        // Normalize: whichever field name was used, put it on req.file
        const files = req.files;
        if (files) {
            req.file = files['media']?.[0] ?? files['image']?.[0] ?? files['file']?.[0];
        }
        next();
    });
};
dotenv.config();
// ── GenLayer client ───────────────────────────────────────────────────────────
// Install: npm install genlayer-js
// Docs: https://docs.genlayer.com/api-references/genlayer-js
import { createClient, createAccount } from 'genlayer-js';
import { studionet } from 'genlayer-js/chains';
import { TransactionStatus } from 'genlayer-js/types';
const CONTRACT = process.env.CONTRACT_ADDRESS || process.env.NEXT_PUBLIC_CONTRACT_ADDRESS || "";
// createAccount() generates an in-memory keypair for server-side signing.
// The private key signs txs on behalf of your backend wallet.
// If you want a persistent wallet, set GENLAYER_PRIVATE_KEY in .env
// and use: createAccount(process.env.GENLAYER_PRIVATE_KEY as `0x${string}`)
const glAccount = createAccount();
// "simulator" points to the hosted GenLayer Studio RPC at studio.genlayer.com
// Switch to testnet chains when you move to production.
const glClient = createClient({
    chain: studionet,
    account: glAccount,
});
// ── Groq Client ───────────────────────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
// ─────────────────────────────────────────────────────────────────────────────
// STEP 1: Vision — describe image using Groq's vision model
// ─────────────────────────────────────────────────────────────────────────────
async function describeImageWithGroq(imageBuffer, mimetype) {
    try {
        const base64Image = imageBuffer.toString('base64');
        const dataUrl = `data:${mimetype};base64,${base64Image}`;
        const response = await groq.chat.completions.create({
            model: 'meta-llama/llama-4-scout-17b-16e-instruct',
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'image_url', image_url: { url: dataUrl } },
                        {
                            type: 'text',
                            text: `You are a visual storytelling analyst. Describe this image in rich, detailed, human-centered language.
Focus on: what is happening, the people/objects/places shown, the mood or atmosphere, any text or symbolic content visible, and the likely context or meaning.
Write 3-5 sentences of flowing descriptive prose. Be specific and accurate.`,
                        },
                    ],
                },
            ],
        });
        const description = response.choices[0]?.message?.content ?? '';
        console.log('🖼️  Vision description:', description.substring(0, 120) + '...');
        return description;
    }
    catch (err) {
        console.warn('  Vision model failed:', err.message);
        return '';
    }
}
// ── Audio Transcription using Groq Whisper ─────────────────────────────
import { toFile } from 'openai'; // Groq SDK is compatible with OpenAI's toFile helper
async function transcribeVideo(videoPath) {
    try {
        console.log('🎙️ Extracting audio for transcription...');
        const audioBuffer = await new Promise((resolve, reject) => {
            const chunks = [];
            ffmpeg(videoPath)
                .outputFormat('mp3')
                .on('error', reject)
                .pipe(new (require('stream').Writable)({
                write(chunk, _encoding, callback) {
                    chunks.push(chunk);
                    callback();
                },
            }))
                .on('finish', () => resolve(Buffer.concat(chunks)));
        });
        // Use Groq's recommended way (via toFile helper)
        const file = await toFile(audioBuffer, 'audio.mp3');
        const response = await groq.audio.transcriptions.create({
            file: file,
            model: 'whisper-large-v3',
            response_format: 'verbose_json',
        });
        console.log('✅ Transcription successful');
        return response.text || '';
    }
    catch (err) {
        console.warn('⚠️ Transcription failed (continuing without audio):', err.message);
        return '';
    }
}
async function extractKeyFrames(videoPath, count = 6) {
    return new Promise((resolve, reject) => {
        const frames = [];
        const tempDir = '/tmp';
        const prefix = `frame-${Date.now()}`;
        const timestamps = [];
        // Evenly spaced timestamps
        for (let i = 1; i <= count; i++) {
            timestamps.push(i * (100 / (count + 1)));
        }
        let completed = 0;
        ffmpeg(videoPath)
            .on('error', (err) => {
            console.error('FFmpeg error:', err);
            reject(err);
        })
            .on('end', async () => {
            // Read all generated frames into buffers
            try {
                for (let i = 1; i <= count; i++) {
                    const filename = `${prefix}-00${i}.jpg`; // ffmpeg naming pattern
                    const filePath = path.join(tempDir, filename);
                    if (await fs.stat(filePath).catch(() => false)) {
                        const buffer = await fs.readFile(filePath);
                        frames.push(buffer);
                        await fs.unlink(filePath).catch(() => { }); // cleanup
                    }
                }
                resolve(frames);
            }
            catch (err) {
                reject(err);
            }
        })
            .screenshots({
            count,
            folder: '/tmp',
            filename: `frame-${Date.now()}-%i.jpg`,
            size: '720x?',
            timemarks: timestamps.map(t => `${t}%`),
        })
            // This is the missing part — read files into buffers
            .on('filenames', (filenames) => {
            // Read each saved frame into buffer
            Promise.all(filenames.map(async (filename) => {
                const fullPath = `/tmp/${filename}`;
                try {
                    const buffer = await import('fs/promises').then(fs => fs.readFile(fullPath));
                    frames.push(buffer);
                    // Optional: cleanup
                    import('fs/promises').then(fs => fs.unlink(fullPath).catch(() => { }));
                }
                catch (e) {
                    console.warn('Failed to read frame:', filename);
                }
            })).then(() => {
                completed = frames.length;
            });
        });
    });
}
ffmpeg.setFfmpegPath(ffmpegInstaller.path);
ffmpeg.setFfprobePath(ffprobeInstaller.path);
async function analyzeVideoWithGroq(videoPath) {
    try {
        console.log('🎥 Extracting frames from video...');
        const frames = await extractKeyFrames(videoPath, 5); // 5 frames is good balance
        console.log('🎙️ Transcribing audio...');
        const transcript = await transcribeVideo(videoPath);
        const descriptions = [];
        for (let i = 0; i < frames.length; i++) {
            console.log(`🖼️ Describing frame ${i + 1}/${frames.length}...`);
            const desc = await describeImageWithGroq(frames[i], 'image/jpeg');
            descriptions.push(desc);
        }
        const prompt = `
Analyze this video based on the following key frame descriptions and audio transcript.

Key Frames:
${descriptions.map((d, i) => `Frame ${i + 1}: ${d}`).join('\n\n')}

${transcript ? `Audio Transcript:\n${transcript}\n` : ''}

Provide a rich, narrative description of the video. Focus on:
- What is happening visually
- Sequence of events
- Mood and atmosphere
- Key actions or messages
`;
        const response = await groq.chat.completions.create({
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            messages: [
                {
                    role: "user",
                    content: [
                        { type: "text", text: prompt }
                    ]
                }
            ],
            temperature: 0.7,
            max_tokens: 900,
        });
        const result = response.choices[0]?.message?.content || "Video description unavailable";
        console.log('✅ Video analysis completed');
        return result;
    }
    catch (err) {
        console.error(" Video analysis failed:", err.message);
        return "Unable to analyze video content.";
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// STEP 2: Semantic analysis — text + optional image description → footprint
// ─────────────────────────────────────────────────────────────────────────────
export async function analyzeMemoryWithGroq(title, content, excerpt, tags, imageDescription) {
    const tagHint = tags.length > 0 ? `User-supplied tags: ${tags.join(', ')}.` : '';
    const imageSection = imageDescription
        ? `\nVisual content description (from attached media):\n"""\n${imageDescription}\n"""\n`
        : '';
    const systemPrompt = `You are a semantic memory analyst for GenLayer Chronicles, a decentralised storytelling platform.
Transform user-submitted memories — which may include text AND visual media — into a rich structured semantic footprint stored on-chain.
When visual content is provided, treat it as primary evidence. Never say content is unclear.
Respond ONLY with a valid JSON object. No markdown fences, no prose outside the JSON.`;
    const userPrompt = `Title: "${title}"
Excerpt: "${excerpt}"
Written content:
"""
${content || '(no written content — rely on visual description below)'}
"""
${imageSection}${tagHint}

Produce a JSON object with EXACTLY these keys:
{
  "core_story": "<2-4 sentence distillation of the central human narrative>",
  "context": "<temporal, geographic, social, or conceptual framing>",
  "emotional_cues": ["<emotion word>", ...],
  "key_entities": ["<person/place/object/concept>", ...],
  "full_text": "<single richly-worded paragraph for on-chain validators — at least 3 sentences>"
}

Rules: emotional_cues 3-6 items, key_entities 2-5 items, full_text minimum 3 sentences.`;
    const completion = await groq.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        temperature: 0.4,
        max_tokens: 1024,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
        ],
    });
    const raw = completion.choices[0]?.message?.content ?? '{}';
    try {
        const cleaned = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return {
            core_story: parsed.core_story ?? '',
            context: parsed.context ?? '',
            emotional_cues: Array.isArray(parsed.emotional_cues) ? parsed.emotional_cues : [],
            key_entities: Array.isArray(parsed.key_entities) ? parsed.key_entities : [],
            full_text: parsed.full_text ?? content,
            image_description: imageDescription ?? '',
        };
    }
    catch {
        console.warn('  Groq returned non-JSON – using fallback');
        return {
            core_story: content.substring(0, 200) || imageDescription?.substring(0, 200) || '',
            context: '',
            emotional_cues: [],
            key_entities: [],
            full_text: content || imageDescription || '',
            image_description: imageDescription ?? '',
        };
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// STEP 3: Submit to GenLayer contract (fire-and-forget, non-blocking)
// ─────────────────────────────────────────────────────────────────────────────
async function submitToGenLayer(params) {
    const { mongoId, mediaUrl, mediaType, title, footprint } = params;
    // Build groq_tags: combine emotional_cues + key_entities, comma-separated, max 5
    const groqTags = [
        ...footprint.emotional_cues,
        ...footprint.key_entities,
    ]
        .slice(0, 5)
        .join(',');
    // writeContract returns a tx hash immediately (before consensus)
    const txHash = await glClient.writeContract({
        address: CONTRACT,
        functionName: 'store_memory',
        args: [
            mongoId,
            mediaUrl,
            mediaType,
            title,
            footprint.core_story,
            footprint.full_text,
            groqTags,
        ],
        value: 0n,
    });
    console.log(`⛓️  GenLayer tx submitted: ${txHash}`);
    // Wait for ACCEPTED status in the background (consensus takes ~30-60s on Studio)
    // We do NOT await this in the request handler — it would time out the HTTP request.
    glClient
        .waitForTransactionReceipt({
        hash: txHash,
        status: TransactionStatus.ACCEPTED,
        retries: 60,
        interval: 5000,
    })
        .then(() => {
        console.log(`✅ GenLayer tx accepted: ${txHash}`);
    })
        .catch((err) => {
        console.warn(`  GenLayer tx wait failed: ${err.message}`);
    });
    return txHash;
}
// ── Cloudinary ────────────────────────────────────────────────────────────────
cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
});
function uploadToCloudinary(buffer, mimetype) {
    const resourceType = mimetype.startsWith('video/') ? 'video' : 'image';
    return new Promise((resolve, reject) => {
        const stream = cloudinary.uploader.upload_stream({
            folder: 'chronicles',
            resource_type: resourceType
        }, (err, result) => {
            if (err || !result)
                return reject(err ?? new Error('Upload failed'));
            resolve(result.secure_url);
        });
        Readable.from(buffer).pipe(stream);
    });
}
async function deleteFromCloudinary(url) {
    try {
        const parts = url.split('/');
        const file = parts[parts.length - 1].split('.')[0];
        const folder = parts[parts.length - 2];
        const publicId = folder === 'chronicles' ? `chronicles/${file}` : file;
        await cloudinary.uploader.destroy(publicId);
    }
    catch (err) {
        console.error('Failed to delete image from Cloudinary:', err);
    }
}
// ── MongoDB ───────────────────────────────────────────────────────────────────
const MONGODB_URI = process.env.MONGODB_URI || process.env.MONGO_URI;
if (!MONGODB_URI) {
    console.error(' No MongoDB URI found in .env');
    process.exit(1);
}
mongoose
    .connect(MONGODB_URI)
    .then(() => console.log('✅ MongoDB connected'))
    .catch(err => console.error(' MongoDB error:', err));
// await StoryModel.updateMany(
//   { mediaType: { $exists: false } },
//   { $set: { mediaType: 'image' } }
// );
// console.log('✅ Existing stories updated with mediaType');
const storySchema = new Schema({
    tag: { type: String, enum: ['Story', 'Moment', 'Milestone', 'Lesson'], default: 'Story' },
    title: { type: String, required: true },
    excerpt: String,
    author: { type: String, required: true },
    likes: { type: Number, default: 0 },
    comments: { type: Number, default: 0 },
    image: String,
    mediaType: {
        type: String,
        enum: ['image', 'video'],
        default: 'image'
    }, // ← Added for video support
    content: { type: String, required: true },
    tags: [String],
    createdAt: { type: Date, default: Date.now },
    semanticFootprint: {
        core_story: { type: String, default: '' },
        context: { type: String, default: '' },
        emotional_cues: { type: [String], default: [] },
        key_entities: { type: [String], default: [] },
        full_text: { type: String, default: '' },
        image_description: { type: String, default: '' },
    },
    genLayerTxHash: { type: String, default: '' },
    onChainConfirmed: { type: Boolean, default: false },
});
const commentSchema = new Schema({
    storyId: { type: mongoose.Schema.Types.ObjectId, ref: 'Story', required: true },
    parentCommentId: { type: mongoose.Schema.Types.ObjectId, ref: 'Comment', default: null },
    author: { type: String, required: true },
    content: { type: String, required: true },
    createdAt: { type: Date, default: Date.now },
});
const StoryModel = model('Story', storySchema);
const CommentModel = model('Comment', commentSchema);
// ── Express ───────────────────────────────────────────────────────────────────
const app = express();
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/tmp'); // or './uploads'
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});
// Allow multiple possible field names
const uploadMiddleware = (fieldName = 'media') => upload.single(fieldName);
app.use((req, res, next) => {
    const allowed = [
        process.env.FRONTEND_URL || 'http://localhost:3000',
        'http://localhost:3000',
        'http://127.0.0.1:3000',
    ];
    const origin = req.headers.origin;
    if (!origin || allowed.includes(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin || '*');
    }
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization');
    if (req.method === 'OPTIONS') {
        res.sendStatus(204);
        return;
    }
    next();
});
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));
function norm(doc) {
    const o = doc.toObject ? doc.toObject() : { ...doc };
    o.id = o._id.toString();
    delete o._id;
    delete o.__v;
    return o;
}
function buildSemanticRelatedQuery(story) {
    const conditions = [];
    if (Array.isArray(story.tags) && story.tags.length) {
        conditions.push({ tags: { $in: story.tags } });
    }
    if (Array.isArray(story.semanticFootprint?.key_entities) && story.semanticFootprint.key_entities.length) {
        conditions.push({ 'semanticFootprint.key_entities': { $in: story.semanticFootprint.key_entities } });
    }
    if (Array.isArray(story.semanticFootprint?.emotional_cues) && story.semanticFootprint.emotional_cues.length) {
        conditions.push({ 'semanticFootprint.emotional_cues': { $in: story.semanticFootprint.emotional_cues } });
    }
    if (conditions.length === 0) {
        return { _id: { $ne: story._id } };
    }
    return {
        _id: { $ne: story._id },
        $or: conditions,
    };
}
async function fetchSemanticFallback(story, limit) {
    const query = buildSemanticRelatedQuery(story);
    return StoryModel.find(query).limit(limit).sort({ createdAt: -1 });
}
// ── Routes ────────────────────────────────────────────────────────────────────
app.get('/health', (_req, res) => res.json({ ok: true, port: process.env.PORT || 3001 }));
// ─────────────────────────────────────────────────────────────────────────────
// POST /api/analyze-memory
// Vision + semantic analysis. Does NOT write to GenLayer or MongoDB.
// Use this when you want the footprint before creating the story record.
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/analyze-memory', flexibleUpload, async (req, res) => {
    try {
        const { title, excerpt, content, tags } = req.body;
        if (!title) {
            res.status(400).json({ error: 'title is required' });
            return;
        }
        const safeTags = typeof tags === 'string'
            ? tags.split(',').map((t) => t.trim()).filter(Boolean)
            : Array.isArray(tags) ? tags : [];
        let visualDescription = '';
        if (req.file) {
            console.log(`📁 File received: ${req.file.originalname} (${req.file.mimetype})`);
            const isVideo = req.file.mimetype.startsWith('video/');
            if (isVideo) {
                visualDescription = await analyzeVideoWithGroq(req.file.path);
            }
            else {
                const imgBuffer = req.file.buffer ?? await fs.readFile(req.file.path);
                visualDescription = await describeImageWithGroq(imgBuffer, req.file.mimetype);
            }
        }
        console.log(`🧠 Groq analyzing: "${title}"`);
        const semanticFootprint = await analyzeMemoryWithGroq(title, content || '', excerpt || (content || '').substring(0, 100), safeTags, visualDescription);
        res.json({ semanticFootprint });
    }
    catch (err) {
        console.error(' Analysis error:', err);
        res.status(500).json({ error: err.message || 'Failed to analyze memory' });
    }
});
// ─────────────────────────────────────────────────────────────────────────────
// POST /api/stories
// Full pipeline: upload image → Groq analysis → save MongoDB → submit GenLayer
// GenLayer submission is fire-and-forget (non-blocking) so HTTP response is fast
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/stories', flexibleUpload, async (req, res) => {
    // upload.single('media')(req, res,)});
    // async (req: Request, res: Response) => {
    try {
        const { title, excerpt, content, author, tag, tags, semanticFootprint: rawFootprint, genLayerTxHash, } = req.body;
        if (!title || !content || !author) {
            res.status(400).json({ error: 'title, content and author are required' });
            return;
        }
        // 1. Upload image
        // === MEDIA HANDLING ===
        let mediaUrl = '';
        let mediaType = 'image';
        let mediaBuffer = null;
        let videoPath = null;
        if (req.file) {
            const isVideo = req.file.mimetype.startsWith('video/');
            mediaType = isVideo ? 'video' : 'image';
            videoPath = req.file.path;
            // diskStorage never populates req.file.buffer — always read from disk
            const fileBuffer = req.file.buffer ?? await fs.readFile(req.file.path);
            mediaBuffer = fileBuffer;
            mediaUrl = await uploadToCloudinary(fileBuffer, req.file.mimetype);
        }
        const safeTags = typeof tags === 'string'
            ? tags.split(',').map((t) => t.trim()).filter(Boolean)
            : Array.isArray(tags) ? tags : [];
        // 2. Parse or generate semantic footprint
        let footprint = null;
        if (rawFootprint) {
            try {
                const parsed = typeof rawFootprint === 'string' ? JSON.parse(rawFootprint) : rawFootprint;
                // Only use pre-computed footprint if it has the required fields
                if (parsed.full_text) {
                    footprint = {
                        core_story: parsed.core_story ?? '',
                        context: parsed.context ?? '',
                        emotional_cues: parsed.emotional_cues ?? [],
                        key_entities: parsed.key_entities ?? [],
                        full_text: parsed.full_text,
                        image_description: parsed.image_description ?? '',
                    };
                }
            }
            catch { /* non-fatal */ }
        }
        if (!footprint) {
            let visualDescription = '';
            if (mediaType === 'image' && mediaBuffer) {
                visualDescription = await describeImageWithGroq(mediaBuffer, req.file.mimetype);
            }
            else if (mediaType === 'video' && videoPath) {
                visualDescription = await analyzeVideoWithGroq(videoPath);
            }
            footprint = await analyzeMemoryWithGroq(title, content, excerpt || content.substring(0, 100), safeTags, visualDescription);
        }
        // 3. Save to MongoDB first — always succeeds even if GenLayer is down
        const story = await new StoryModel({
            tag: tag || 'Story',
            title,
            author,
            content,
            excerpt: excerpt || content.substring(0, 100),
            image: mediaUrl || undefined,
            tags: safeTags,
            mediaType,
            semanticFootprint: {
                core_story: footprint.core_story,
                context: footprint.context,
                emotional_cues: footprint.emotional_cues,
                key_entities: footprint.key_entities,
                full_text: footprint.full_text,
                image_description: footprint.image_description,
                footprint,
            },
            genLayerTxHash: genLayerTxHash || '',
            onChainConfirmed: false,
        }).save();
        console.log('✅ Story saved:', story._id);
        // 4. Submit to GenLayer asynchronously — does NOT block the HTTP response
        //    The tx hash is written back to MongoDB once available.
        if (CONTRACT) {
            submitToGenLayer({
                mongoId: story._id.toString(),
                mediaUrl,
                mediaType,
                title,
                footprint,
            })
                .then(txHash => {
                console.log('✅ GenLayer tx hash received:', txHash); // add this
                return StoryModel.findByIdAndUpdate(story._id, { genLayerTxHash: txHash });
            })
                .catch(err => console.error('❌ GenLayer submission failed:', err.message, err.stack));
        }
        res.status(201).json(norm(story));
    }
    catch (err) {
        console.error(' Save error:', err);
        res.status(500).json({ error: err.message || 'Failed to save story' });
    }
});
// ─────────────────────────────────────────────────────────────────────────────
// PATCH /api/stories/:id/onchain
// Called manually or by a webhook once GenLayer tx is confirmed
// ─────────────────────────────────────────────────────────────────────────────
app.patch('/api/stories/:id/onchain', async (req, res) => {
    try {
        const { genLayerTxHash } = req.body;
        const updated = await StoryModel.findByIdAndUpdate(req.params.id, { genLayerTxHash, onChainConfirmed: true }, { new: true });
        if (!updated) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        res.json(norm(updated));
    }
    catch (err) {
        res.status(500).json({ error: err.message || 'Failed to update on-chain status' });
    }
});
// ─────────────────────────────────────────────────────────────────────────────
// GET /api/stories/:id/related
// Uses GenLayer contract's find_related() → enriches with full MongoDB records
// Falls back to simple tag-based lookup if GenLayer is unavailable
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/stories/:id/related', async (req, res) => {
    try {
        const story = await StoryModel.findById(req.params.id);
        if (!story) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        const limit = Math.min(20, Math.max(1, Number(req.query.limit ?? 6)));
        const respondWithFallback = async () => {
            const related = await fetchSemanticFallback(story, limit);
            res.json(related.map(norm));
        };
        if (!CONTRACT) {
            return respondWithFallback();
        }
        try {
            const rawResult = await glClient.readContract({
                address: CONTRACT,
                functionName: 'find_related',
                args: [story._id.toString(), limit],
            });
            const relatedOnChain = JSON.parse(rawResult);
            const mongoIds = relatedOnChain.map(r => r.mongo_id).filter(Boolean);
            if (mongoIds.length === 0) {
                return respondWithFallback();
            }
            const stories = await StoryModel.find({ _id: { $in: mongoIds } });
            if (stories.length === 0) {
                return respondWithFallback();
            }
            return res.json(stories.map(norm));
        }
        catch (glErr) {
            console.warn('  GenLayer read failed, falling back to semantic related search:', glErr?.message || glErr);
            return respondWithFallback();
        }
    }
    catch (err) {
        console.error('Related stories error:', err);
        res.status(500).json({ error: 'Failed to fetch related stories' });
    }
});
// ── Standard CRUD routes ──────────────────────────────────────────────────────
app.get('/api/stories', async (req, res) => {
    try {
        const query = {};
        if (req.query.tag && req.query.tag !== 'All')
            query.tag = req.query.tag;
        if (req.query.search) {
            query.$or = [
                { title: { $regex: req.query.search, $options: 'i' } },
                { excerpt: { $regex: req.query.search, $options: 'i' } },
            ];
        }
        const stories = await StoryModel.find(query).sort({ createdAt: -1 });
        res.json(stories.map(norm));
    }
    catch {
        res.status(500).json({ error: 'Failed to fetch stories' });
    }
});
app.get('/api/stories/:id', async (req, res) => {
    try {
        const story = await StoryModel.findById(req.params.id);
        if (!story) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        const comments = await CommentModel.find({ storyId: req.params.id }).sort({ createdAt: 1 });
        const normalized = comments.map(norm);
        const commentMap = new Map();
        const roots = [];
        normalized.forEach(comment => {
            comment.replies = [];
            commentMap.set(comment.id, comment);
        });
        normalized.forEach(comment => {
            if (comment.parentCommentId && commentMap.has(comment.parentCommentId)) {
                commentMap.get(comment.parentCommentId).replies.push(comment);
            }
            else {
                roots.push(comment);
            }
        });
        const obj = norm(story);
        obj.comment = roots;
        res.json(obj);
    }
    catch (err) {
        console.error('Fetch story error:', err);
        res.status(500).json({ error: 'Failed to fetch story' });
    }
});
app.put('/api/stories/:id', upload.single('media'), async (req, res) => {
    try {
        const existing = await StoryModel.findById(req.params.id);
        if (!existing) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        let imageUrl = existing.image ?? '';
        if (req.file) {
            if (imageUrl)
                await deleteFromCloudinary(imageUrl);
            imageUrl = await uploadToCloudinary(req.file.buffer, req.file.mimetype);
        }
        const { title, excerpt, content, author, tag, tags } = req.body;
        const safeTags = typeof tags === 'string'
            ? tags.split(',').map((t) => t.trim()).filter(Boolean)
            : Array.isArray(tags) ? tags : [];
        const updated = await StoryModel.findByIdAndUpdate(req.params.id, { tag, title, excerpt, content, author, image: imageUrl, tags: safeTags }, { new: true });
        res.json(norm(updated));
    }
    catch (err) {
        res.status(500).json({ error: err.message || 'Failed to update story' });
    }
});
app.delete('/api/stories/:id', async (req, res) => {
    try {
        const story = await StoryModel.findById(req.params.id);
        if (!story) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        if (story.image)
            await deleteFromCloudinary(story.image);
        await StoryModel.findByIdAndDelete(req.params.id);
        await CommentModel.deleteMany({ storyId: req.params.id });
        res.json({ deleted: true });
    }
    catch (err) {
        res.status(500).json({ error: err.message || 'Failed to delete story' });
    }
});
app.post('/api/stories/:id/like', async (req, res) => {
    try {
        const story = await StoryModel.findByIdAndUpdate(req.params.id, { $inc: { likes: 1 } }, { new: true });
        if (!story) {
            res.status(404).json({ error: 'Not found' });
            return;
        }
        res.json(norm(story));
    }
    catch {
        res.status(500).json({ error: 'Failed to like story' });
    }
});
app.get('/api/stories/:id/comments', async (req, res) => {
    try {
        const comments = await CommentModel
            .find({ storyId: req.params.id })
            .sort({ createdAt: -1 });
        res.json(comments.map(norm));
    }
    catch {
        res.status(500).json({ error: 'Failed to fetch comments' });
    }
});
app.post('/api/stories/:id/comments', async (req, res) => {
    try {
        const { author, content, parentCommentId } = req.body;
        if (!author || !content) {
            res.status(400).json({ error: 'author and content required' });
            return;
        }
        const comment = await new CommentModel({
            storyId: req.params.id,
            parentCommentId: parentCommentId || null,
            author,
            content,
        }).save();
        await StoryModel.findByIdAndUpdate(req.params.id, { $inc: { comments: 1 } });
        res.status(201).json(norm(comment));
    }
    catch (err) {
        console.error('Add comment error:', err);
        res.status(500).json({ error: 'Failed to add comment' });
    }
});
app.get('/api/stats', async (_req, res) => {
    try {
        const [totalStories, totalComments, likesAgg] = await Promise.all([
            StoryModel.countDocuments(),
            CommentModel.countDocuments(),
            StoryModel.aggregate([{ $group: { _id: null, total: { $sum: '$likes' } } }]),
        ]);
        res.json({
            totalStories,
            totalCommunityMembers: 1,
            totalLikes: likesAgg[0]?.total ?? 0,
            totalComments,
        });
    }
    catch {
        res.status(500).json({ error: 'Failed to fetch stats' });
    }
});
// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || '3001', 10);
app.listen(PORT, () => {
    console.log(` Server on http://localhost:${PORT}`);
    console.log(`   Contract: ${CONTRACT || '(NEXT_PUBLIC_CONTRACT_ADDRESS not set)'}`);
});
