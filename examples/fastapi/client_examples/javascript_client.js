/**
 * JavaScript/Node.js 클라이언트 예제
 * 
 * pyhub-llm FastAPI 서비스를 사용하는 JavaScript 클라이언트 예제입니다.
 * 
 * 설치 방법:
 * npm install axios
 * 
 * 실행 방법:
 * node javascript_client.js
 */

const axios = require('axios');

class PyHubLLMClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null, timeout = 30000) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = timeout;
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
        
        // Axios 인스턴스 생성
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers: this.headers
        });
    }
    
    async chat(message, options = {}) {
        const {
            model = 'gpt-4o-mini',
            systemPrompt = null,
            temperature = null,
            protected = false
        } = options;
        
        const endpoint = protected ? '/api/chat' : '/chat';
        
        const payload = {
            message,
            model
        };
        
        if (systemPrompt) payload.system_prompt = systemPrompt;
        if (temperature !== null) payload.temperature = temperature;
        
        try {
            const response = await this.client.post(endpoint, payload);
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    async batch(messages, options = {}) {
        const {
            model = 'gpt-4o-mini',
            maxParallel = 3,
            historyMode = 'independent',
            protected = false
        } = options;
        
        const endpoint = protected ? '/api/batch' : '/batch';
        
        const payload = {
            messages,
            model,
            max_parallel: maxParallel,
            history_mode: historyMode
        };
        
        try {
            // 배치 처리는 시간이 더 걸릴 수 있으므로 타임아웃 연장
            const response = await this.client.post(endpoint, payload, {
                timeout: this.timeout * messages.length
            });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    async translate(text, targetLanguage, options = {}) {
        const {
            sourceLanguage = null,
            model = 'gpt-4o-mini'
        } = options;
        
        const payload = {
            text,
            target_language: targetLanguage,
            model
        };
        
        if (sourceLanguage) payload.source_language = sourceLanguage;
        
        try {
            const response = await this.client.post('/api/translate', payload);
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    async summarize(text, options = {}) {
        const {
            maxLength = 200,
            language = 'ko',
            model = 'gpt-4o-mini'
        } = options;
        
        const payload = {
            text,
            max_length: maxLength,
            language,
            model
        };
        
        try {
            const response = await this.client.post('/api/summarize', payload);
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    async getSupportedLanguages() {
        try {
            const response = await this.client.get('/api/supported-languages');
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    async healthCheck() {
        try {
            const response = await this.client.get('/health', { timeout: 5000 });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }
    
    handleError(error) {
        if (error.response) {
            // 서버가 응답했지만 에러 상태 코드
            const { status, data } = error.response;
            return new Error(`HTTP ${status}: ${data.detail || data.message || 'Unknown error'}`);
        } else if (error.request) {
            // 요청이 만들어졌지만 응답을 받지 못함
            return new Error('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
        } else {
            // 요청을 설정하는 중에 오류 발생
            return new Error(`요청 오류: ${error.message}`);
        }
    }
}

// =============================================================================
// 사용 예제
// =============================================================================

async function basicExamples() {
    console.log('🚀 기본 사용 예제');
    
    // 클라이언트 생성 (인증 없이)
    const client = new PyHubLLMClient();
    
    try {
        // 헬스체크
        const health = await client.healthCheck();
        console.log(`✅ 서비스 상태: ${health.status}`);
        
        // 단일 채팅
        console.log('\n💬 단일 채팅 예제:');
        const chatResult = await client.chat(
            'FastAPI와 pyhub-llm의 조합에 대해 간단히 설명해주세요.'
        );
        console.log(`응답: ${chatResult.response.substring(0, 100)}...`);
        
        // 배치 처리
        console.log('\n📦 배치 처리 예제:');
        const questions = [
            'JavaScript의 주요 특징 3가지는?',
            'Node.js를 사용하는 이유는?',
            'REST API 설계 원칙은?'
        ];
        
        const batchResult = await client.batch(questions);
        console.log(`✅ ${batchResult.success_count}/${batchResult.total_count} 성공`);
        console.log(`⏱️ 실행 시간: ${batchResult.execution_time.toFixed(2)}초`);
        
        batchResult.responses.forEach((response, i) => {
            console.log(`Q${i+1}: ${questions[i]}`);
            console.log(`A${i+1}: ${response.response.substring(0, 80)}...`);
            console.log('');
        });
        
    } catch (error) {
        console.error('❌ 오류:', error.message);
    }
}

async function authenticatedExamples() {
    console.log('🔐 인증 예제');
    
    // API 키가 필요한 클라이언트
    const apiKey = 'demo-key-12345'; // 실제 환경에서는 환경변수 사용
    const client = new PyHubLLMClient('http://localhost:8000', apiKey);
    
    try {
        // 보호된 채팅
        console.log('\n🛡️ 보호된 채팅:');
        const protectedResult = await client.chat(
            '인증이 필요한 보안 채팅입니다!',
            { protected: true }
        );
        console.log(`응답: ${protectedResult.response.substring(0, 100)}...`);
        
        // 번역 서비스
        console.log('\n🌍 번역 서비스:');
        const translation = await client.translate(
            'Hello, how are you today?',
            'ko'
        );
        console.log(`원문: ${translation.original_text}`);
        console.log(`번역: ${translation.translated_text}`);
        
        // 요약 서비스
        console.log('\n📝 요약 서비스:');
        const longText = `
        Node.js는 Chrome V8 JavaScript 엔진으로 빌드된 JavaScript 런타임입니다.
        이벤트 기반의 비차단 I/O 모델을 사용하여 가볍고 효율적입니다.
        npm 생태계는 세계에서 가장 큰 오픈 소스 라이브러리 생태계이기도 합니다.
        FastAPI와 pyhub-llm을 JavaScript/Node.js에서 사용하면 강력한
        LLM 기반 애플리케이션을 구축할 수 있습니다.
        `;
        
        const summary = await client.summarize(longText.trim(), {
            maxLength: 100
        });
        console.log(`원본 길이: ${summary.original_length}자`);
        console.log(`요약 길이: ${summary.summary_length}자`);
        console.log(`압축률: ${(summary.compression_ratio * 100).toFixed(1)}%`);
        console.log(`요약: ${summary.summary}`);
        
        // 지원 언어 목록
        console.log('\n🌐 지원 언어:');
        const languages = await client.getSupportedLanguages();
        if (languages.premium_user) {
            console.log(`✨ 프리미엄 사용자: ${languages.total_count}개 언어 지원`);
        } else {
            console.log(`📚 기본 사용자: ${languages.total_count}개 언어 지원`);
        }
        
    } catch (error) {
        console.error('❌ 오류:', error.message);
    }
}

async function advancedBatchExamples() {
    console.log('⚡ 고급 배치 처리 예제');
    
    const client = new PyHubLLMClient();
    
    try {
        // 순차적 대화
        console.log('\n🔗 순차적 대화:');
        const sequentialQuestions = [
            '리액트(React)에 대해 설명해주세요.',
            '이것의 주요 개념들을 알려주세요.',
            '실제 프로젝트에서 어떻게 사용하나요?',
            '다른 프레임워크와 비교했을 때 장점은?'
        ];
        
        const sequentialResult = await client.batch(sequentialQuestions, {
            historyMode: 'sequential'
        });
        
        console.log('📈 순차적 대화 결과:');
        sequentialResult.responses.forEach((response, i) => {
            console.log(`\n단계 ${i+1}: ${sequentialQuestions[i]}`);
            console.log(`응답: ${response.response.substring(0, 120)}...`);
        });
        
        // 공유 컨텍스트
        console.log('\n🤝 공유 컨텍스트 배치:');
        const contextQuestions = [
            'Node.js의 주요 특징은?',
            'Express.js와의 차이점은?',
            '성능 최적화 방법은?'
        ];
        
        const sharedResult = await client.batch(contextQuestions, {
            historyMode: 'shared'
        });
        
        console.log('🌟 공유 컨텍스트 결과:');
        sharedResult.responses.forEach((response, i) => {
            console.log(`\nQ${i+1}: ${contextQuestions[i]}`);
            console.log(`A${i+1}: ${response.response.substring(0, 100)}...`);
        });
        
    } catch (error) {
        console.error('❌ 오류:', error.message);
    }
}

async function main() {
    console.log('🌟 pyhub-llm FastAPI JavaScript 클라이언트 예제');
    console.log('='.repeat(50));
    
    try {
        await basicExamples();
        console.log('\n' + '='.repeat(50));
        
        await authenticatedExamples();
        console.log('\n' + '='.repeat(50));
        
        await advancedBatchExamples();
        
    } catch (error) {
        if (error.message.includes('서버에 연결할 수 없습니다')) {
            console.error('❌ 서버에 연결할 수 없습니다.');
            console.log('💡 먼저 FastAPI 서버를 실행해주세요:');
            console.log('   python main.py  또는  python advanced.py');
        } else {
            console.error('❌ 예상치 못한 오류:', error.message);
        }
    }
    
    console.log('\n✅ 모든 예제 완료!');
}

// Node.js 환경에서 실행
if (require.main === module) {
    main().catch(console.error);
}

module.exports = PyHubLLMClient;