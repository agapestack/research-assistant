<script lang="ts">
	import { onMount } from 'svelte';
	import {
		queryStream,
		getFollowups,
		getModels,
		getStats,
		healthCheck,
		type Source,
		type ModelsResponse,
		type StatsResponse
	} from '$lib/api';
	import { theme } from '$lib/stores/theme';
	import SourceCard from '$lib/components/SourceCard.svelte';
	import SearchTab from '$lib/components/Search.svelte';
	import IndexTab from '$lib/components/Index.svelte';

	interface Message {
		id: string;
		role: 'user' | 'assistant';
		content: string;
		sources?: Source[];
		model?: string;
		followups?: string[];
	}

	type Tab = 'chat' | 'search' | 'index';

	let activeTab: Tab = $state('chat');
	let messages: Message[] = $state([]);
	let input = $state('');
	let loading = $state(false);
	let loadingStage = $state<'retrieving' | 'generating' | null>(null);
	let error = $state('');
	let models: ModelsResponse | null = $state(null);
	let selectedModel = $state('');
	let stats: StatsResponse | null = $state(null);
	let healthy = $state(false);
	let k = $state(5);
	let hoveredSource = $state<number | null>(null);
	let sourcesExpanded = $state<Record<string, boolean>>({});

	onMount(async () => {
		theme.init();
		try {
			[models, stats, healthy] = await Promise.all([getModels(), getStats(), healthCheck()]);
			if (models) selectedModel = models.default;
		} catch (e) {
			console.error('Failed to initialize', e);
		}
	});

	function generateId() {
		return Math.random().toString(36).substring(2, 9);
	}

	async function handleSubmit(question?: string) {
		const q = question || input.trim();
		if (!q || loading) return;

		input = '';
		error = '';
		loading = true;
		loadingStage = 'retrieving';

		const userMsg: Message = { id: generateId(), role: 'user', content: q };
		const assistantId = generateId();
		const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', sources: [] };

		messages = [...messages, userMsg, assistantMsg];

		try {
			for await (const event of queryStream(q, k, selectedModel || undefined)) {
				if (event.type === 'sources') {
					loadingStage = 'generating';
					messages = messages.map((m) =>
						m.id === assistantId ? { ...m, sources: event.sources, model: event.model } : m
					);
				} else if (event.type === 'chunk') {
					messages = messages.map((m) =>
						m.id === assistantId ? { ...m, content: m.content + event.content } : m
					);
				} else if (event.type === 'done') {
					const finalMsg = messages.find((m) => m.id === assistantId);
					if (finalMsg) {
						try {
							const followups = await getFollowups(q, finalMsg.content);
							messages = messages.map((m) =>
								m.id === assistantId ? { ...m, followups } : m
							);
						} catch {
							// Followups are optional
						}
					}
				}
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get response';
			messages = messages.filter((m) => m.id !== assistantId);
		} finally {
			loading = false;
			loadingStage = null;
			// Refresh stats after query
			stats = await getStats().catch(() => stats);
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSubmit();
		}
	}

	function clearChat() {
		messages = [];
		error = '';
	}

	function copyAnswer(content: string) {
		navigator.clipboard.writeText(content);
	}

	function toggleSources(msgId: string) {
		sourcesExpanded = { ...sourcesExpanded, [msgId]: !sourcesExpanded[msgId] };
	}

	function renderWithCitations(text: string, sources: Source[]): string {
		return text.replace(/\[(\d+)\]/g, (match, num) => {
			const idx = parseInt(num) - 1;
			const source = sources[idx];
			if (source) {
				return `<button class="citation" data-id="${num}">[${num}]</button>`;
			}
			return match;
		});
	}
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors">
	<!-- Header -->
	<header class="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
		<div class="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
			<div class="flex items-center gap-6">
				<h1 class="text-lg font-semibold">Research Assistant</h1>
				<nav class="flex gap-1">
					{#each [{ id: 'chat', label: 'Chat' }, { id: 'search', label: 'Search' }, { id: 'index', label: 'Index' }] as tab}
						<button
							onclick={() => (activeTab = tab.id as Tab)}
							class="px-3 py-1.5 text-sm font-medium rounded-md transition-colors
								{activeTab === tab.id
								? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
								: 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}"
						>
							{tab.label}
						</button>
					{/each}
				</nav>
			</div>
			<div class="flex items-center gap-3">
				<div class="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
					<span class="w-2 h-2 rounded-full {healthy ? 'bg-green-500' : 'bg-red-500'}"></span>
					{#if stats}
						<span>{stats.total_documents.toLocaleString()} docs</span>
					{/if}
				</div>
				<button
					onclick={() => theme.toggle()}
					class="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
					aria-label="Toggle theme"
				>
					{#if $theme === 'dark'}
						<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
						</svg>
					{:else}
						<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
						</svg>
					{/if}
				</button>
			</div>
		</div>
	</header>

	<!-- Main Content -->
	<main class="max-w-5xl mx-auto px-4 py-6">
		{#if activeTab === 'chat'}
			<div class="space-y-6">
				<!-- Settings Bar -->
				<div class="flex items-center gap-4 text-sm">
					<div class="flex items-center gap-2">
						<label for="model" class="text-gray-500 dark:text-gray-400">Model:</label>
						<select
							id="model"
							bind:value={selectedModel}
							class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						>
							{#if models}
								{#each Object.entries(models.available) as [id, desc]}
									<option value={id} title={desc}>{id}</option>
								{/each}
							{/if}
						</select>
					</div>
					<div class="flex items-center gap-2">
						<label for="k" class="text-gray-500 dark:text-gray-400">Sources:</label>
						<input
							id="k"
							type="number"
							min="1"
							max="20"
							bind:value={k}
							class="w-16 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					{#if messages.length > 0}
						<button
							onclick={clearChat}
							class="ml-auto text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
						>
							Clear chat
						</button>
					{/if}
				</div>

				<!-- Messages -->
				<div class="space-y-8">
					{#if messages.length === 0 && !loading}
						<div class="text-center py-20">
							<div class="text-4xl mb-4">📚</div>
							<h2 class="text-xl font-medium text-gray-700 dark:text-gray-300 mb-2">
								Ask about your research papers
							</h2>
							<p class="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
								Query your indexed academic papers using natural language. Answers include inline citations you can hover to preview.
							</p>
						</div>
					{/if}

					{#each messages as message (message.id)}
						{#if message.role === 'user'}
							<div class="flex justify-end">
								<div class="bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2 max-w-[80%]">
									{message.content}
								</div>
							</div>
						{:else}
							<div class="space-y-4">
								<!-- Sources (collapsible) -->
								{#if message.sources && message.sources.length > 0}
									<div>
										<button
											onclick={() => toggleSources(message.id)}
											class="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
										>
											<svg
												class="w-4 h-4 transition-transform {sourcesExpanded[message.id] ? 'rotate-90' : ''}"
												fill="none"
												viewBox="0 0 24 24"
												stroke="currentColor"
											>
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
											</svg>
											{message.sources.length} sources
										</button>
										{#if sourcesExpanded[message.id]}
											<div class="mt-3 grid gap-2 sm:grid-cols-2">
												{#each message.sources as source}
													<SourceCard {source} compact={true} />
												{/each}
											</div>
										{/if}
									</div>
								{/if}

								<!-- Answer -->
								<div class="prose prose-gray dark:prose-invert max-w-none">
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										class="whitespace-pre-wrap"
										onmouseenter={(e) => {
											const target = e.target as HTMLElement;
											if (target.classList.contains('citation')) {
												hoveredSource = parseInt(target.dataset.id || '0');
											}
										}}
										onmouseleave={() => (hoveredSource = null)}
									>
										{@html renderWithCitations(message.content, message.sources || [])}
									</div>

									<!-- Hovered source preview -->
									{#if hoveredSource && message.sources}
										{@const source = message.sources[hoveredSource - 1]}
										{#if source}
											<div class="fixed z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 max-w-sm text-sm" style="bottom: 100px; right: 20px;">
												<div class="font-medium text-blue-600 dark:text-blue-400">[{hoveredSource}] {source.title}</div>
												{#if source.content}
													<p class="mt-1 text-gray-600 dark:text-gray-300 text-xs line-clamp-3">{source.content}</p>
												{/if}
											</div>
										{/if}
									{/if}
								</div>

								<!-- Actions -->
								<div class="flex items-center gap-4 text-sm">
									{#if message.model}
										<span class="text-gray-400 dark:text-gray-500">via {message.model}</span>
									{/if}
									<button
										onclick={() => copyAnswer(message.content)}
										class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
									>
										Copy
									</button>
								</div>

								<!-- Follow-up questions -->
								{#if message.followups && message.followups.length > 0}
									<div class="flex flex-wrap gap-2">
										{#each message.followups as followup}
											<button
												onclick={() => handleSubmit(followup)}
												class="text-sm px-3 py-1.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full text-gray-700 dark:text-gray-300 transition-colors"
											>
												{followup}
											</button>
										{/each}
									</div>
								{/if}
							</div>
						{/if}
					{/each}

					<!-- Loading indicator -->
					{#if loading}
						<div class="flex items-center gap-3 text-gray-500 dark:text-gray-400">
							<div class="flex gap-1">
								<span class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
								<span class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
								<span class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
							</div>
							<span class="text-sm">
								{loadingStage === 'retrieving' ? 'Searching papers...' : 'Generating answer...'}
							</span>
						</div>
					{/if}
				</div>

				<!-- Error -->
				{#if error}
					<div class="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm rounded-lg">
						{error}
					</div>
				{/if}

				<!-- Input -->
				<div class="sticky bottom-0 pt-4 pb-6 bg-gradient-to-t from-gray-50 dark:from-gray-950 from-80%">
					<form onsubmit={(e) => { e.preventDefault(); handleSubmit(); }} class="relative">
						<input
							type="text"
							bind:value={input}
							onkeydown={handleKeydown}
							placeholder="Ask a question about your papers..."
							class="w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
							disabled={loading}
						/>
						<button
							type="submit"
							disabled={loading || !input.trim()}
							class="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
							aria-label="Send message"
						>
							<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
							</svg>
						</button>
					</form>
				</div>
			</div>
		{:else if activeTab === 'search'}
			<SearchTab />
		{:else if activeTab === 'index'}
			<IndexTab onstatsupdate={async () => { stats = await getStats().catch(() => stats); }} />
		{/if}
	</main>
</div>

<style>
	:global(.citation) {
		color: #2563eb;
		font-weight: 500;
		cursor: pointer;
	}
	:global(.citation:hover) {
		text-decoration: underline;
	}
	:global(.dark .citation) {
		color: #60a5fa;
	}
	:global(.prose) {
		line-height: 1.625;
	}
</style>
