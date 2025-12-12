<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<title>AI Research Assistant</title>
</svelte:head>

{@render children()}
